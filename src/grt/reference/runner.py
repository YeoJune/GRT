"""Native author RMT path: raw labels/mask -> wrapper.loss -> backward.

Uses the existing train CLI, without the common logits-only model adapter.
"""
import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoXConfig, get_linear_schedule_with_warmup
from grt.checkpoint import seed_all, capture_rng, restore_rng
from grt.config import save_config
from grt.logger import metadata,write_json
from grt.reference.modeling_gpt_neox import GPTNeoXForCausalLM
from grt.reference.language_modeling import MemoryCell, RecurrentWrapper
from grt.reference.data import ARDataset, make_collator

COMMIT = "24cbb9aed62a5748c4045fd928f7f59899f03b24"

def write_result_files(root,name,result):
    """Keep JSON and an indented TXT copy readable in the Drive preview."""
    content=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    for suffix in ('json','txt'):
        (root/f'{name}.{suffix}').write_text(content,encoding='utf-8')

def append_text_metrics(stage_dir,row):
    validation=row['validation'];fixed=validation['fixed_length']
    content=(f"Step {row['step']}\n"
        f"  Train: loss={row['train/loss']:.6f}, lr={row['train/lr']:.8g}\n"
        f"  Random-length validation: loss={validation['loss']:.6f}, exact_match={validation['exact_match']:.2%}\n"
        f"  Fixed-length validation:  loss={fixed['loss']:.6f}, exact_match={fixed['exact_match']:.2%}\n"
        f"  Training time={row['train/seconds']:.2f}s, peak VRAM={row['train/peak_gpu_memory_bytes']/2**30:.2f} GiB\n\n")
    with (stage_dir/'metrics.txt').open('a',encoding='utf-8') as f:f.write(content)

def make_model(cfg):
    m=cfg.model
    config=GPTNeoXConfig(vocab_size=m.vocab_size,hidden_size=m.d_model,
        num_hidden_layers=m.alu.num_layers,num_attention_heads=m.alu.nhead,
        intermediate_size=m.alu.d_ff,max_position_embeddings=2048,
        bos_token_id=101,eos_token_id=102,hidden_act='gelu',rotary_pct=.25,
        rotary_emb_base=10000,attention_dropout=m.alu.dropout,hidden_dropout=m.alu.dropout,
        initializer_range=.02,layer_norm_eps=1e-5,use_cache=True,
        tie_word_embeddings=False,use_parallel_residual=True)
    return RecurrentWrapper(MemoryCell(GPTNeoXForCausalLM(config),m.num_registers,wrap_pos=False),
        segment_size=m.segment_len,max_n_segments=cfg.data.train_segments,
        k2=cfg.data.train_segments,segment_alignment='left')

def make_dataset(cfg,split,pairs):
    size=getattr(cfg.data,{'train':'train_samples','validation':'validation_samples','test':'test_samples'}[split])
    offset={'train':0,'validation':1,'test':2}[split]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.data.data_seed+offset)
        return ARDataset(cfg.data.key_size,cfg.data.value_size,pairs,size)

def make_loader(cfg,dataset,training=False,vary=None):
    if vary is None:vary=cfg.data.vary_n_pairs
    args=SimpleNamespace(vary_n_segments=vary,value_size=cfg.data.value_size)
    batch=cfg.training.batch_size*cfg.training.grad_accum_steps if training else cfg.evaluation.batch_size
    # Original loader preserves dataset order. An independent generator keeps iterator creation
    # from consuming the model/data RNG, including when resuming midway through an epoch.
    return DataLoader(dataset,batch_size=batch,collate_fn=make_collator(args),num_workers=0,
                      generator=torch.Generator().manual_seed(cfg.training.model_seed))

def forward_batch(model,batch,device):
    return model(**{k:batch[k].to(device) for k in ('input_ids','attention_mask','labels','labels_mask')})

@torch.no_grad()
def evaluate(model,cfg,dataset,device,*,vary=False):
    from grt.checkpoint import evaluation_context
    ce=correct=count=0
    with evaluation_context(model):
        for batch in make_loader(cfg,dataset,vary=vary):
            output=forward_batch(model,batch,device)
            n=len(batch['input_ids']);ce+=output.loss.item()*n;count+=n
            generated=model.generate(batch['input_ids_generate'].to(device),
                attention_mask=batch['attention_mask_generate'].to(device),
                max_new_tokens=cfg.data.value_size+1,pad_token_id=0,eos_token_id=102)
            target=batch['labels'][:,-cfg.data.value_size-1:].to(device)
            if generated.shape[1]>=target.shape[1]:
                correct+=generated[:,-target.shape[1]:].eq(target).all(1).sum().item()
    return {'loss':ce/count,'exact_match':correct/count,'samples':count}

def run(cfg,device):
    if cfg.training.mixed_precision!='fp32':
        raise ValueError('The author reference path currently requires fp32')
    if (cfg.training.optimizer,cfg.training.scheduler,cfg.training.grad_clip_type)!=('adamw','linear','value'):
        raise ValueError('Author reference requires AdamW, linear schedule and value clipping')
    root=Path(cfg.run.output_dir);root.mkdir(parents=True,exist_ok=True)
    stored=root/'reference_config.json'
    if stored.exists() and json.loads(stored.read_text())!=asdict(cfg):
        raise ValueError('Reference configuration changed; use a new output directory')
    if not stored.exists() and any(root.iterdir()):
        raise ValueError('Nonempty reference output directory without configuration')
    stored.write_text(json.dumps(asdict(cfg),indent=2));save_config(cfg,root/'resolved_config.yaml')
    stages=cfg.training.curriculum or [SimpleNamespace(num_pairs=cfg.data.train_segments-1,
        key_size=cfg.data.key_size,max_steps=cfg.training.max_steps)]
    results=[];previous=None;started=time.perf_counter()
    old_summary=root/'summary.json'
    prior_wall=json.loads(old_summary.read_text()).get('wall_seconds',0) if old_summary.exists() else 0
    for index,stage in enumerate(stages):
        current=copy.deepcopy(cfg);current.data.key_size=stage.key_size
        current.data.train_segments=stage.num_pairs+1
        current.model.segment_len=stage.key_size+current.data.value_size+2
        current.training.max_steps=stage.max_steps
        current.training.warmup_steps=stage.max_steps//10
        current.training.scheduler_steps=stage.max_steps*2
        current.training.curriculum=[]
        current.data.eval_segments=[stage.num_pairs+1]
        stage_dir=root/f'stage_{index+1:02d}_pairs{stage.num_pairs}_key{stage.key_size}'
        stage_dir.mkdir(exist_ok=True);save_config(current,stage_dir/'resolved_config.yaml')
        seed_all(current.training.model_seed)
        model=make_model(current).to(device)
        provenance=metadata(model,current)
        provenance['reference_source']=json.loads(Path(__file__).with_name('SOURCE.json').read_text())
        provenance['native_labels_and_loss']=True
        if not (stage_dir/'metadata.json').exists():write_json(stage_dir/'metadata.json',provenance)
        optimizer=torch.optim.AdamW(model.parameters(),lr=current.training.lr,
            weight_decay=current.training.weight_decay,betas=(.9,current.training.adam_beta2))
        scheduler=get_linear_schedule_with_warmup(optimizer,stage.max_steps//10,stage.max_steps*2)
        train=make_dataset(current,'train',stage.num_pairs)
        valid=make_dataset(current,'validation',stage.num_pairs)
        loader=make_loader(current,train,training=True)
        step=epoch_batch=0;best=-1.;last_validation=None;done=False
        training_seconds=0.;peak_memory=0
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        path=stage_dir/'last.pt'
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
            step=state['step'];epoch_batch=state['epoch_batch'];best=state['best_exact_match']
            done=state['done'];last_validation=state['validation']
            training_seconds=state.get('training_seconds',0.)
            peak_memory=state.get('peak_training_gpu_memory_bytes',0)
        elif previous is not None:
            model.load_state_dict(previous)
        iterator=iter(loader)
        # Rebuild the loader cursor without changing the checkpoint RNG used by the collator.
        if epoch_batch:
            rng=capture_rng()
            for _ in range(epoch_batch):next(iterator)
            restore_rng(rng)
        def save(target,done_flag):
            temporary=target.with_suffix('.tmp')
            torch.save({'format':'author_rmt/1','reference_commit':COMMIT,'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'rng':capture_rng(),'step':step,'epoch_batch':epoch_batch,
                'best_exact_match':best,'validation':last_validation,'done':done_flag,
                'training_seconds':training_seconds,'peak_training_gpu_memory_bytes':peak_memory,
                'config':asdict(current)},temporary)
            temporary.replace(target)
        model.train()
        while step<stage.max_steps and not done:
            if cfg.training.max_seconds is not None and prior_wall+time.perf_counter()-started>=cfg.training.max_seconds:break
            if device.type=='cuda':torch.cuda.synchronize(device)
            update_started=time.perf_counter()
            optimizer.zero_grad(set_to_none=True);loss_total=0.
            try:effective_batch=next(iterator)
            except StopIteration:iterator=iter(loader);epoch_batch=0;effective_batch=next(iterator)
            epoch_batch+=1
            for start in range(0,len(effective_batch['input_ids']),current.training.batch_size):
                batch={k:v[start:start+current.training.batch_size] for k,v in effective_batch.items()}
                loss=forward_batch(model,batch,device).loss/current.training.grad_accum_steps
                loss.backward();loss_total+=loss.item()
            torch.nn.utils.clip_grad_value_(model.parameters(),current.training.grad_clip)
            optimizer.step();scheduler.step();step+=1
            if device.type=='cuda':
                torch.cuda.synchronize(device)
                peak_memory=max(peak_memory,torch.cuda.max_memory_allocated(device))
            training_seconds+=time.perf_counter()-update_started
            if step%current.evaluation.every_steps==0 or step==stage.max_steps:
                # Original random-length validation plus an explicit maximum-length functional check.
                last_validation=evaluate(model,current,valid,device,vary=current.data.vary_n_pairs)
                fixed=evaluate(model,current,valid,device,vary=False)
                last_validation['fixed_length']=fixed
                score=fixed['exact_match'] if current.training.stop_on_convergence else last_validation['exact_match']
                if score>best:
                    best=score;save(stage_dir/'best.pt',False)
                done=current.training.stop_on_convergence and fixed['exact_match']>=current.training.convergence_exact_match
                row={'step':step,'train/loss':loss_total,'train/lr':optimizer.param_groups[0]['lr'],
                    'train/seconds':training_seconds,'train/peak_gpu_memory_bytes':peak_memory,
                    'validation':last_validation}
                with (stage_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                append_text_metrics(stage_dir,row)
                print(f'Author RMT {stage.num_pairs} pairs step {step}: loss={loss_total:.4f}, fixed EM={fixed["exact_match"]:.3f}',flush=True)
                save(path,done)
        if last_validation is None:
            last_validation=evaluate(model,current,valid,device,vary=False)
            last_validation['fixed_length']=dict(last_validation)
            best=last_validation['exact_match'];save(stage_dir/'best.pt',done)
        done=done or step>=stage.max_steps;save(path,done)
        selected=torch.load(stage_dir/'best.pt',map_location='cpu',weights_only=False)
        previous=selected['model']
        converged=last_validation['fixed_length']['exact_match']>=current.training.convergence_exact_match
        results.append({'num_pairs':stage.num_pairs,'key_size':stage.key_size,'step':step,
            'completed':done,'converged':converged,'checkpoint':str(stage_dir/'best.pt'),
            'training_seconds':training_seconds,'peak_training_gpu_memory_bytes':peak_memory,
            'stop_reason':'converged' if current.training.stop_on_convergence and converged else ('update_budget' if done else 'time_budget'),
            'validation':last_validation})
        status='completed' if done and index==len(stages)-1 else 'interrupted'
        if current.training.stop_on_convergence and done and not converged:status='stage_not_converged'
        write_result_files(root,'summary',{'reference_commit':COMMIT,'status':status,'stages':results,
            'wall_seconds':prior_wall+time.perf_counter()-started})
        if not done or (current.training.stop_on_convergence and not converged):break
    model.load_state_dict(previous)
    test=make_dataset(current,'test',stage.num_pairs)
    evaluation=evaluate(model,current,test,device,vary=False)
    write_result_files(root,'evaluation',{'checkpoint':str(stage_dir/'best.pt'),
        'checkpoint_step':selected['step'],'num_pairs':stage.num_pairs,**evaluation})
    return results
