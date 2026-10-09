from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from grt.config import load_config, validate_config
from grt.curriculum import stage_config
from grt.data import generate_balanced_remember, remember_context_capacity, make_loader
from grt.evaluator import evaluate_remember_generation

ROOT = Path(__file__).resolve().parents[1]


def config(path):
    return validate_config(load_config([ROOT/'configs/rmt_paper_remember.yaml',
                                       ROOT/'configs/rmt_remember_functional.yaml'], output_dir=path))


def canonical(batch):
    facts = batch['input_ids'][:-4].reshape(-1,4)
    return tuple(sorted((row[0],row[2]) for row in facts.tolist()))


def test_single_fact_exhaustive_split_partition():
    partitions = []
    for split in ('train','validation','test'):
        capacity = remember_context_capacity(1,split)[1]
        partition = {canonical(generate_balanced_remember(split,i,1)) for i in range(capacity)}
        assert len(partition) == capacity
        partitions.append(partition)
    assert [len(p) for p in partitions] == [192,32,32]
    assert len(set.union(*partitions)) == 256
    assert not any(partitions[i] & partitions[j] for i,j in ((0,1),(0,2),(1,2)))


@pytest.mark.parametrize('pairs',[2,3,5,8])
def test_every_query_uses_same_context_and_distinct_values(pairs):
    batches = [generate_balanced_remember('test',17*pairs+i,pairs) for i in range(pairs)]
    facts = batches[0]['input_ids'][:-4].reshape(pairs,4)
    assert len(set(facts[:,0].tolist())) == len(set(facts[:,2].tolist())) == pairs
    for position,batch in enumerate(batches):
        assert torch.equal(batch['input_ids'][:-4],facts.flatten())
        assert batch['input_ids'][-4] == facts[position,0]
        assert batch['labels'][-3] == facts[position,2]
        assert batch['labels'][-2] == 102


def test_multifact_contexts_are_unique_and_split_disjoint():
    partitions = []
    for split in ('train','validation','test'):
        contexts = {canonical(generate_balanced_remember(split,i*2,2)) for i in range(512)}
        assert len(contexts) == 512
        partitions.append(contexts)
    assert len(set.union(*partitions)) == 1536


def test_functional_sizes_fit_mapping_space_and_complete_contexts(tmp_path):
    cfg = config(tmp_path)
    stages = [stage_config(cfg,i) for i in range(3)]
    assert [(s.data.train_samples,s.data.validation_samples,s.data.test_samples) for s in stages] == [
        (192,32,32),(24576,1024,1024),(24576,1023,1023)]
    for pairs,stage in zip((1,2,3),stages):
        assert not stage.data.vary_n_pairs
        assert stage.data.train_samples % pairs == stage.data.validation_samples % pairs == 0
        assert stage.training.batch_size == 1024 and stage.training.grad_accum_steps == 1
    # A wrapped training sample must preserve its source and query position.
    loader = make_loader(stages[1].data,'train',3,4,4,start_id=stages[1].data.train_samples)
    batch = next(iter(loader))
    assert torch.equal(batch['input_ids'][0],generate_balanced_remember('train',0,2)['input_ids'])


def test_position_shortcut_fails_all_queries_even_with_perfect_eos(tmp_path):
    cfg = stage_config(config(tmp_path),1)
    class LastValue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.cfg = SimpleNamespace(segment_len=4)
        def generate_answer(self,prompt,max_new_tokens):
            facts = prompt[:,:-2].reshape(prompt.shape[0],-1,4)
            values = facts[:,-1,2]
            return torch.stack([values,torch.full_like(values,102)],dim=1)
    loader = make_loader(cfg.data,'test',3,12,3)  # Context query groups cross batch boundaries.
    result = evaluate_remember_generation(LastValue(),loader,1,balanced_queries=True)
    assert result['value_exact_match'] == .5 and result['token_accuracy'] == .75
    assert result['all_queries_exact_match'] == 0 and result['contexts'] == 6
    assert result['query_position_accuracy'] == [0.,1.]
    with pytest.raises(ValueError,match='complete context'):
        evaluate_remember_generation(LastValue(),make_loader(cfg.data,'test',3,3,3),1,balanced_queries=True)


def test_convergence_requires_all_query_success(tmp_path,monkeypatch):
    from grt.models.factory import create_model
    from grt.trainer import Trainer
    cfg = stage_config(config(tmp_path),1)
    cfg.model.d_model = 32
    cfg.model.num_registers = 4
    cfg.model.alu.num_layers = 2
    cfg.model.alu.d_ff = 32
    cfg.data.validation_samples = cfg.evaluation.autoregressive_samples = 4
    cfg.evaluation.batch_size = 2
    generation = {'token_accuracy':.9975,'exact_match':.995,'value_exact_match':.995,
                  'all_queries_exact_match':.985,'query_position_accuracy':[.995,.995],
                  'samples':4,'contexts':2}
    monkeypatch.setattr('grt.trainer.evaluate_remember_generation',lambda *a,**kw:dict(generation))
    trainer = Trainer(create_model(cfg.model),cfg)
    trainer.validate()
    assert not trainer.converged
    generation['all_queries_exact_match'] = .99
    trainer.validate()
    assert trainer.converged
