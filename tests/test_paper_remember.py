from dataclasses import asdict
from pathlib import Path
import copy
import json
import pytest
import torch
from grt.checkpoint import seed_all, load_checkpoint, restore_model
from grt.config import load_config, validate_config, ModelConfig, ALUConfig, CurriculumStage
from grt.curriculum import stage_config, run_curriculum
from grt.data import generate_remember, collate, make_loader
from grt.evaluator import evaluate_remember_generation
from grt.metrics import masked_ce
from grt.models.factory import create_model
from grt.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(path):
    cfg = load_config(ROOT / "configs/rmt_paper_remember.yaml", output_dir=path)
    cfg.model.d_model = 32
    cfg.model.num_registers = 4
    cfg.model.alu.num_layers = 2
    cfg.model.alu.d_ff = 32
    cfg.training.curriculum = []
    cfg.training.batch_size = 2
    cfg.training.max_steps = 3
    cfg.training.warmup_steps = 0
    cfg.training.max_seconds = None
    cfg.evaluation.batch_size = 2
    cfg.evaluation.every_steps = 1
    cfg.evaluation.autoregressive_samples = 4
    cfg.evaluation.warmup = 0
    cfg.evaluation.iterations = 1
    cfg.data.validation_samples = 4
    cfg.data.test_samples = 4
    cfg.data.train_segments = 4
    cfg.data.eval_segments = [4]
    return validate_config(cfg)


@pytest.mark.parametrize("pairs,key_size", [(1,1),(5,1),(20,2)])
def test_remember_unique_keys_query_and_next_token_labels(pairs, key_size):
    batch = generate_remember("test", 23, pairs, key_size)
    n = key_size + 3
    segments = batch['input_ids'].reshape(pairs+1, n)
    facts, query = segments[:-1], segments[-1]
    assert len(set(map(tuple, facts[:, :key_size].tolist()))) == pairs
    matching = (facts[:, :key_size] == query[:key_size]).all(dim=1)
    assert matching.sum() == 1
    assert query[-2] == facts[matching][0,-2]
    assert (facts[:, key_size] == 100).all() and query[key_size] == 101
    assert (segments[:, -1] == 102).all()
    assert batch['labels'][batch['labels'] != -100].tolist() == query[-2:].tolist()
    assert batch['labels'][-3] == query[-2] and batch['labels'][-2] == 102
    assert (batch['labels'][:-3] == -100).all() and batch['labels'][-1] == -100
    assert torch.equal(batch['input_ids'], generate_remember('test',23,pairs,key_size)['input_ids'])


def test_vary_pairs_is_per_batch_and_validation_is_fixed(tmp_path):
    cfg = tiny_config(tmp_path)
    batches = list(make_loader(cfg.data,'train',6,40,4))
    assert len({b['input_ids'].shape[1] for b in batches}) > 1
    assert all(b['input_ids'].shape[0] == 4 for b in batches)
    assert all(b['input_ids'].shape[1] == 24 for b in make_loader(cfg.data,'validation',6,12,4))


def test_author_backbone_wrapper_logits_memory_gradients_and_generation():
    fixture = torch.load(ROOT / 'tests/fixtures/neox_rmt_reference.pt', weights_only=True)
    cfg = ModelConfig(name='rmt', vocab_size=128, segment_len=4, num_registers=4,
        d_model=32, tie_word_embeddings=False, rmt_backbone='gpt_neox',
        alu=ALUConfig(num_layers=2,nhead=4,d_ff=32,dropout=0))
    model = create_model(cfg)
    model.load_state_dict(fixture['state'])
    ids = fixture['input_ids']
    logits = model(ids).logits
    torch.testing.assert_close(logits, fixture['logits'], rtol=1e-5, atol=1e-6)
    labels = torch.full_like(ids, -100)
    labels[:, -3:-1] = ids[:, -2:]
    loss = masked_ce(logits,labels)
    torch.testing.assert_close(loss,fixture['loss'])
    loss.backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad,fixture['grads'][name], rtol=2e-5, atol=1e-6)
    model.eval()
    with torch.no_grad():
        memory = model.initial_memory(ids.shape[0])
        for index, segment in enumerate(ids.split(4,dim=1)):
            _, memory = model.forward_segment(segment,memory)
            torch.testing.assert_close(memory,fixture['memories'][index],rtol=1e-5,atol=1e-6)
        assert torch.equal(model.generate_answer(ids[:,:-2],2),fixture['generated'])


def test_future_answer_cannot_change_query_logits_and_memory_gradient_reaches_facts(tmp_path):
    model = create_model(tiny_config(tmp_path).model)
    batch = collate([generate_remember('test',i,3) for i in range(2)])
    changed = batch['input_ids'].clone()
    changed[:,-2:] = 15
    original = model(batch['input_ids']).logits
    torch.testing.assert_close(original[:,-3],model(changed).logits[:,-3],rtol=0,atol=0)
    memory = model.initial_memory(2)
    _, first_memory = model.forward_segment(batch['input_ids'][:,:4],memory)
    first_memory.retain_grad()
    memory = first_memory
    for segment in batch['input_ids'][:,4:].split(4,dim=1):
        logits,memory = model.forward_segment(segment,memory)
    masked_ce(logits,batch['labels'][:,-4:]).backward()
    assert first_memory.grad is not None and first_memory.grad.norm() > 0


def test_generation_evaluator_only_passes_prompt(tmp_path):
    cfg = tiny_config(tmp_path)
    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
        def generate_answer(self,prompt,max_new_tokens):
            assert prompt.shape[1] % 4 == 2 and max_new_tokens == 2
            facts = prompt[:,:-2].reshape(prompt.shape[0],-1,4)
            matches = facts[:,:,0] == prompt[:,-2,None]
            values = facts[:,:,2][matches]
            return torch.stack([values,torch.full_like(values,102)],dim=1)
    result = evaluate_remember_generation(Oracle(),make_loader(cfg.data,'test',4,4,2),1)
    assert result['exact_match'] == result['value_exact_match'] == 1


def test_remember_resume_matches_next_update(tmp_path):
    cfg = tiny_config(tmp_path/'full')
    seed_all(123)
    full = Trainer(create_model(cfg.model),cfg)
    full.train()
    other = copy.deepcopy(cfg)
    other.run.output_dir = str(tmp_path/'split')
    seed_all(123)
    split = Trainer(create_model(other.model),other)
    split.train(stop_after=1)
    state = load_checkpoint(tmp_path/'split/last.pt')
    model,saved_cfg = restore_model(state)
    resumed = Trainer(model,saved_cfg)
    resumed.resume(state)
    resumed.train()
    for name,p in full.model.state_dict().items():
        torch.testing.assert_close(p,resumed.model.state_dict()[name],rtol=0,atol=0)


def test_curriculum_stops_on_failed_gate_and_resumes(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.training.curriculum = [CurriculumStage(1,1,1),CurriculumStage(2,1,1)]
    cfg.training.convergence_exact_match = 1
    cfg.training.max_steps = 1
    cfg.data.train_segments = 2
    cfg.data.eval_segments = [3]
    records = run_curriculum(cfg,'cpu')
    assert len(records) == 1 and not records[0]['converged']
    assert not (tmp_path/'stage_02_pairs2_key1').exists()
    resumed = run_curriculum(cfg,'cpu')
    assert resumed[0]['global_step'] == records[0]['global_step'] == 1
    assert json.loads((tmp_path/'summary.json').read_text())['status'] == 'stage_not_converged'


def test_stage_key_length_changes_without_changing_weights(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.training.curriculum = [CurriculumStage(1,1,10),CurriculumStage(20,2,10)]
    cfg.data.eval_segments = [21]
    first,last = [stage_config(cfg,i) for i in (0,1)]
    assert first.model.segment_len == 4 and last.model.segment_len == 5
    assert first.training.scheduler_steps == last.training.scheduler_steps == 20
    target = create_model(last.model)
    target.load_state_dict(create_model(first.model).state_dict())
    out = target(next(iter(make_loader(last.data,'test',21,2,2)))['input_ids'])
    assert out.logits.shape == (2,105,128)


def test_curriculum_transfers_trained_weights_and_resets_optimizer(tmp_path,monkeypatch):
    cfg = tiny_config(tmp_path)
    cfg.training.curriculum = [CurriculumStage(1,1,1),CurriculumStage(2,1,1)]
    cfg.data.train_segments = 2
    cfg.data.eval_segments = [3]
    # Supply a successful generation metric to exercise the stage transition.
    monkeypatch.setattr('grt.trainer.evaluate_remember_generation', lambda *a,**kw:
        {'token_accuracy':1.,'exact_match':1.,'value_exact_match':1.,'samples':4})
    original_train = Trainer.train
    checked = []
    def inspect_transfer(self,**kwargs):
        if self.cfg.data.train_segments == 3:
            previous = load_checkpoint(tmp_path/'stage_01_pairs1_key1/converged.pt')
            for name,value in self.model.state_dict().items():
                torch.testing.assert_close(value,previous['model'][name],rtol=0,atol=0)
            assert not self.optimizer.state and self.global_step == 0
            checked.append(True)
        return original_train(self,**kwargs)
    monkeypatch.setattr(Trainer,'train',inspect_transfer)
    records = run_curriculum(cfg,'cpu')
    assert checked == [True] and len(records) == 2 and all(r['converged'] for r in records)


def test_time_budget_stops_before_optimizer_update(tmp_path):
    cfg = tiny_config(tmp_path)
    trainer = Trainer(create_model(cfg.model),cfg)
    progress = trainer.train(time_limit_seconds=0)
    assert progress['global_step'] == 0 and progress['stop_reason'] == 'time_budget'
    assert (tmp_path/'last.pt').exists() and (tmp_path/'best.pt').exists()
