"""Regenerate the small golden fixture from the author's pinned code.

python tests/fixtures/generate_neox_reference.py /path/to/author-repo
Checkout 24cbb9aed62a5748c4045fd928f7f59899f03b24 first.
Requires transformers==4.44.2. No training or network downloads.
"""
import ast
import subprocess
import sys
import types
from pathlib import Path
import torch
from transformers import GPTNeoXConfig

COMMIT = "24cbb9aed62a5748c4045fd928f7f59899f03b24"


def source(repo, relative):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{COMMIT}:{relative}"], text=True)


def main(repo):
    torch.set_num_threads(1)
    tree = ast.parse(source(repo, "base_models/modeling_gpt_neox.py"))
    # This optional adapter is disabled; removing only its import avoids unrelated dependencies.
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "adapters")]
    backbone_module = types.ModuleType("golden_reference_neox")
    sys.modules[backbone_module.__name__] = backbone_module
    exec(compile(tree, "author/modeling_gpt_neox.py", "exec"), backbone_module.__dict__)
    memory_module = types.ModuleType("golden_reference_rmt")
    exec(source(repo, "modeling_rmt/language_modeling.py"), memory_module.__dict__)
    torch.manual_seed(123)
    config = GPTNeoXConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=32, max_position_embeddings=2048,
        rotary_pct=0.25, hidden_act="gelu", hidden_dropout=0.0, attention_dropout=0.0,
        bos_token_id=101, eos_token_id=102, tie_word_embeddings=False, use_cache=False)
    backbone = backbone_module.GPTNeoXForCausalLM(config)
    cell = memory_module.MemoryCell(backbone, 4, wrap_pos=False)
    wrapper = memory_module.RecurrentWrapper(cell, segment_size=4, max_n_segments=4, k2=4)
    ids = torch.tensor([[1,100,7,102, 4,100,3,102, 9,100,12,102, 4,101,3,102],
                        [2,100,6,102, 5,100,8,102, 8,100,11,102, 2,101,6,102]])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, -3:] = True  # Original loss mask indexes predictors, including GEN.
    output = wrapper(ids, attention_mask=torch.ones_like(ids), labels=ids, labels_mask=mask,
                     output_hidden_states=True)
    output.loss.backward()
    state = {"backbone."+k: v.detach().clone() for k,v in backbone.state_dict().items()
             if not k.endswith("rotary_emb.inv_freq")}
    state["mem0"] = cell.memory.detach().clone()
    grads = {"backbone."+k: v.grad.detach().clone() for k,v in backbone.named_parameters()}
    grads["mem0"] = cell.memory.grad.detach().clone()
    memory = None
    memories = []
    wrapper.eval()
    with torch.no_grad():
        for segment in ids.split(4, dim=1):
            _, memory = cell(segment, memory_state=memory)
            memories.append(memory.clone())
        # The 4.31 prepare_inputs_for_generation omits use_cache from model kwargs.
        # Set the backbone flag explicitly for its native cached generation path.
        backbone.config.use_cache = True
        generated = wrapper.generate(ids[:, :-2], attention_mask=torch.ones_like(ids[:, :-2]),
                                     max_new_tokens=2, eos_token_id=None, pad_token_id=0, use_cache=True)
    fixture = {"reference_commit": COMMIT, "transformers": "4.44.2", "state": state,
               "input_ids": ids, "logits": output.logits.detach(), "loss": output.loss.detach(),
               "grads": grads, "memories": torch.stack(memories), "generated": generated}
    path = Path(__file__).with_name("neox_rmt_reference.pt")
    torch.save(fixture, path)
    print(path, path.stat().st_size)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
