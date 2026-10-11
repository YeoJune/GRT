"""New architecture contracts: masked retrieval, causal options, and stream resume."""
import copy
from pathlib import Path
import torch
import pytest
from grt.config import validate_config
from grt.experiments import load_experiment
from grt.models.factory import create_model
from grt.data import make_dataset, make_loader, forward_batch
from grt.trainer import run
from grt.wikitext import WikiTextAdapter
from scripts.prepare_wikitext import prepare


def config(tmp_path, name='grt', mode='encoder'):
    cfg = load_experiment(task='remember', model=name, runtime='cpu', steps=4, output_dir=tmp_path,
        overrides={'model': {'attention_mode': mode, 'd_model': 32, 'num_registers': 4,
                            'alu': {'num_layers': 2, 'nhead': 4, 'd_ff': 64}},
                   'data': {'options': {'objective': 'parallel' if mode == 'encoder' else 'next_token'}}})
    if name == 'grt':
        cfg.model.router.mlp_hidden = 32
    cfg.model.attention_mode = mode
    if mode == 'decoder':
        cfg.data.options = {}
    return cfg


@pytest.mark.parametrize('name', ['rmt','grt'])
def test_parallel_query_has_no_answer_and_one_stack_per_segment(tmp_path, name):
    cfg = config(tmp_path, name)
    model = create_model(cfg).eval()
    batch = next(iter(make_loader(cfg,make_dataset(cfg,'validation',2),vary=False)))
    assert (batch['input_ids'][:,-2:] == 103).all()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    out = forward_batch(model,batch,'cpu')
    hook.remove()
    assert len(calls) == 3
    target = batch['labels'][batch['labels_mask']]
    expected = torch.nn.functional.cross_entropy(out.logits[batch['labels_mask']],target)
    torch.testing.assert_close(out.loss,expected)
    changed = dict(batch, labels=(batch['labels']+1)%128)
    other = forward_batch(model,changed,'cpu')
    torch.testing.assert_close(out.logits,other.logits,rtol=0,atol=0)
    generated = model.generate(batch['input_ids_generate'],max_new_tokens=2)
    assert torch.equal(generated,out.logits[:,-2:].argmax(-1))
    out.loss.backward()
    assert model.initial_memory.grad.abs().sum()>0
    if name=='grt':
        assert model.router.mlp_r[-1].weight.grad is None
        assert model.router.mlp_w[-1].weight.grad.abs().sum()>0
        with torch.no_grad():
            model.router.mlp_r[-1].weight.add_(100)
        torch.testing.assert_close(out.logits,forward_batch(model,batch,'cpu').logits,rtol=0,atol=0)
        _,trace=model.forward_with_trace(**{k:batch[k] for k in ['input_ids','labels','labels_mask','attention_mask']})
        assert len(trace.records['w_gates'])==3
    assert model.backbone.embedding.weight is model.backbone.lm_head.weight
    restored=create_model(cfg)
    restored.load_state_dict(model.state_dict())
    assert restored.backbone.embedding.weight is restored.backbone.lm_head.weight


@pytest.mark.parametrize('name,source', [('rmt','intermediate'),('grt','fixed_state')])
def test_decoder_causality_including_register_path(tmp_path,name,source):
    torch.manual_seed(19)
    cfg=config(tmp_path,name,'decoder')
    cfg.model.register_read_source=source
    cfg.model.segment_len=8
    model=create_model(cfg).eval()
    ids=torch.tensor([[1,2,3,4,5,6]])
    state=torch.randn(1,cfg.model.num_registers,cfg.model.d_model)
    altered=ids.clone();altered[:,3:]=torch.tensor([[9,10,11]])
    before=model(ids,memory_state=state).logits[:,:3]
    after=model(altered,memory_state=state).logits[:,:3]
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    # Shorter prefixes must also preserve logits with nonzero incoming memory.
    prefix=model(ids[:,:3],memory_state=state).logits
    torch.testing.assert_close(before,prefix,rtol=1e-5,atol=1e-7)
    assert not torch.allclose(model(ids).memory_state,model(altered).memory_state)


def test_intermediate_decoder_keeps_encoder_register_interaction(tmp_path):
    torch.manual_seed(19)
    cfg=config(tmp_path,mode='decoder');cfg.model.segment_len=8
    model=create_model(cfg).eval()
    ids=torch.tensor([[1,2,3,4,5,6]])
    altered=ids.clone();altered[:,3:]=torch.tensor([[9,10,11]])
    assert not torch.allclose(model(ids).logits[:,:3],model(altered).logits[:,:3])


def test_read_switch_and_fixed_state_candidate_gradient(tmp_path):
    cfg=config(tmp_path);cfg.model.register_read_source='fixed_state'
    cfg.model.register.conditional_read=True
    model=create_model(cfg)
    batch=next(iter(make_loader(cfg,make_dataset(cfg,'validation',1),vary=False)))
    out=forward_batch(model,batch,'cpu');out.loss.backward()
    assert model.router.mlp_r[-1].weight.grad.abs().sum()>0


@pytest.mark.parametrize('name',['grt','rmt'])
def test_remember_encoder_training_and_resume(tmp_path,name,monkeypatch):
    cfg=config(tmp_path,name)
    from grt import trainer
    save = trainer.atomic_save
    early_path = tmp_path/'step2.pt'
    def snapshot(state, path):
        save(state,path)
        if state.get('step') == 2 and Path(path).name == 'last.pt':
            save(state,early_path)
    monkeypatch.setattr(trainer,'atomic_save',snapshot)
    run(cfg,torch.device('cpu'))
    directory=tmp_path/'stage_01_pairs1_key1'
    full=torch.load(directory/'last.pt',weights_only=False)
    earlier=torch.load(early_path,weights_only=False)
    assert earlier['step'] == 2
    torch.save(earlier,directory/'last.pt')
    run(cfg,torch.device('cpu'))
    resumed=torch.load(directory/'last.pt',weights_only=False)
    for key,value in full['model'].items():
        torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)


def stream_config(tmp_path, name='grt'):
    source=tmp_path/'source';source.mkdir()
    for split in ['train','valid','test']:
        (source/f'wiki.{split}.tokens').write_text('one two three <unk>\n'*60)
    prepared=tmp_path/'tokens';manifest=prepare(source,prepared,0)
    cfg=config(tmp_path/'run',name=name,mode='decoder')
    cfg.model.register_read_source='fixed_state'
    cfg.model.vocab_size=manifest['vocab_size'];cfg.model.segment_len=4
    cfg.data.name='wikitext103';cfg.data.options={'path':str(prepared)}
    cfg.training.curriculum=[];cfg.training.bptt_segments=1
    cfg.evaluation.every_steps=2
    return cfg


@pytest.mark.parametrize('name',['grt','rmt'])
def test_word_stream_boundary_loss_and_exact_resume(tmp_path,monkeypatch,name):
    cfg=stream_config(tmp_path,name);validate_config(cfg)
    cfg.model.alu.dropout=0.1
    from grt import stream_training
    save=stream_training.atomic_save
    early_path=tmp_path/'step2.pt'
    def snapshot(state,path):
        save(state,path)
        if state.get('step')==2 and Path(path).name=='last.pt':
            save(state,early_path)
    monkeypatch.setattr(stream_training,'atomic_save',snapshot)
    run(cfg,torch.device('cpu'))
    root=Path(cfg.run.output_dir)
    full=torch.load(root/'last.pt',weights_only=False)
    early=torch.load(early_path,weights_only=False)
    assert full['target_tokens_seen']==4*4*4
    assert early['step']==2
    torch.save(early,root/'last.pt')
    run(cfg,torch.device('cpu'))
    resumed=torch.load(root/'last.pt',weights_only=False)
    for key,value in full['model'].items():
        torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
    adapter=WikiTextAdapter()
    result=adapter.evaluate(create_model(cfg),cfg,adapter.dataset(cfg,'test'),'cpu')
    assert result['target_tokens']==296
    assert result['score_kind']=='autoregressive'
    assert result['loss']==pytest.approx(result['nll_sum']/result['target_tokens'])


def test_replay_gradients_end_at_detached_state(tmp_path):
    cfg=config(tmp_path,mode='decoder')
    cfg.model.register_read_source='fixed_state'
    model=create_model(cfg)
    old=model(torch.tensor([[1,2,3,4]]))
    old.memory_state.retain_grad()
    replay_embeddings=[]
    def capture(module,args,out):
        out.retain_grad()
        replay_embeddings.append(out)
    hook=model.backbone.embedding.register_forward_hook(capture)
    ids=torch.tensor([[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]])
    targets=torch.full_like(ids,-100);targets[:,-4:]=torch.tensor([[18,19,20,21]])
    out=model(ids,target_ids=targets,memory_state=old.memory_state.detach())
    hook.remove()
    out.loss.backward()
    assert old.memory_state.grad is None
    assert len(replay_embeddings)==4
    assert all(x.grad is not None and x.grad.abs().sum()>0 for x in replay_embeddings)


def test_stream_preflight_preserves_effective_batch(tmp_path):
    from grt.benchmark import benchmark
    from grt.distributed import DistributedContext
    cfg=stream_config(tmp_path)
    cfg.training.batch_size=2;cfg.training.grad_accum_steps=2;cfg.training.global_batch_size=4
    report=benchmark(cfg,DistributedContext('cpu'),steps=1)
    assert report['global_batch_size']==4
    assert report['replay_tokens_per_stream']==8
    assert report['target_tokens_per_update']==16
    assert report['update_seconds']>0
    assert not (Path(cfg.run.output_dir)/'last.pt').exists()


@pytest.mark.parametrize('name',['grt','rmt'])
def test_stream_distributed_cli_and_checkpoint_state(tmp_path,name):
    from dataclasses import asdict
    import json
    import yaml
    from test_distributed import invoke
    cfg=stream_config(tmp_path,name)
    cfg.training.batch_size=1;cfg.training.grad_accum_steps=2;cfg.training.global_batch_size=4
    path=tmp_path/'cfg.yaml';path.write_text(yaml.safe_dump(asdict(cfg)))
    root=Path(cfg.run.output_dir)
    invoke(path,root)
    saved=torch.load(root/'last.pt',weights_only=False)
    assert saved['world_size']==2
    assert len(saved['stream_by_rank'])==2
    assert saved['target_tokens_seen']==64
    assert json.loads((root/'evaluation.json').read_text())['target_tokens']==296
    invoke(path,root,benchmark=True)
    report=json.loads((root/'benchmark.json').read_text())
    assert report['global_batch_size']==report['microbatch_per_gpu']*report['accumulation']*2==4


def test_base_head_41_and_size_presets(tmp_path):
    for name in ['grt','rmt']:
        cfg=load_experiment(task='wikitext103', model=name, size='base', output_dir=tmp_path)
        validate_config(cfg)
        cfg.model.alu.num_layers=1;cfg.model.vocab_size=16;cfg.model.num_registers=2
        model=create_model(cfg)
        assert model(torch.tensor([[1,2,3]])).logits.shape==(1,3,16)


def test_backbone_weights_and_sampling_match_across_models(tmp_path):
    models=[];batches=[]
    for name in ['rmt','grt']:
        cfg=config(tmp_path,name)
        torch.manual_seed(54)
        models.append(create_model(cfg))
        batches.append(next(iter(make_loader(cfg,make_dataset(cfg,'train',2),vary=True))))
    for key,value in models[0].backbone.state_dict().items():
        torch.testing.assert_close(value,models[1].backbone.state_dict()[key],rtol=0,atol=0)
    for key,value in batches[0].items():
        assert torch.equal(value,batches[1][key])


def test_multiple_value_slots_and_query_gradients(tmp_path):
    cfg=config(tmp_path)
    cfg.data.key_size=2;cfg.data.value_size=2;cfg.model.segment_len=6
    validate_config(cfg)
    model=create_model(cfg)
    batch=next(iter(make_loader(cfg,make_dataset(cfg,'validation',2),vary=False)))
    assert (batch['input_ids'][:,-3:]==103).all()
    assert batch['labels_mask'].sum()==batch['input_ids'].shape[0]*3
    out=forward_batch(model,batch,'cpu')
    out.loss.backward()
    assert model.initial_memory.grad.abs().sum()>0
    assert model.generate(batch['input_ids'],max_new_tokens=3).shape==(4,3)


def test_wikitext_encoder_is_not_silently_trained_as_next_token(tmp_path):
    cfg=stream_config(tmp_path)
    cfg.model.attention_mode='encoder'
    with pytest.raises(ValueError,match='encoder objective is unspecified'):
        validate_config(cfg)
