"""Remember data bodies extracted from author commit 24cbb9; no task changes."""
import torch
from tqdm.auto import tqdm
NUM_SYMBOLS = 16
rewrite_setting = False

def generate_pairs(key_size, value_size, num_pairs, num_samples):
    keys = torch.empty((num_samples, num_pairs, key_size))

    if not rewrite_setting:
        for i in tqdm(range(num_samples)):
            key = torch.randperm(NUM_SYMBOLS ** key_size)[:num_pairs]
            for j in range(key_size):
                keys[i, :, j] = key % NUM_SYMBOLS
                key //= NUM_SYMBOLS
    else:
        keys = torch.randint(0, NUM_SYMBOLS, (num_samples, num_pairs, key_size))
    
    values = torch.randint(0, NUM_SYMBOLS, (num_samples, num_pairs, value_size))

    # if vary_n_pairs:
    #     keys_list = []
    #     values_list = []
    #     for key in keys:
    #         n = torch.randint(1, len(key)+1)
    #         keys_list.append(key[-n:])
    #         values_list.append(values[-n:])
    #     keys = keys_list
    #     values = values_list
    
    # keys = torch.randint(0, NUM_SYMBOLS, (num_pairs * 2, key_size))
    # keys[:, 0] = torch.randint(1, NUM_SYMBOLS, (num_pairs * 2, ))
    
    # unique = keys.unique(dim=0)
    # delta_pairs = num_pairs - unique.shape[0]
    # if delta_pairs > 0:
    #     print('got unique')
    #     return generate_pairs(key_size, value_size, num_pairs)

    # selected_ids = torch.randperm(unique.shape[0])[:num_pairs]
    # keys = unique[selected_ids]

    # values[:, 0] = torch.randint(1, NUM_SYMBOLS, (num_pairs, ))
    return keys, values

class ARDataset:
    def __init__(self, key_size, value_size, sample_len=1, num_samples=20_000):
        self.sample_len = sample_len
        self.keys, self.values = generate_pairs(key_size, value_size, sample_len, num_samples)
        # self.keys = keys.reshape(num_samples, -1)
        # self.values = values.reshape(num_samples, -1)
        if not rewrite_setting:
            self.target_key_inds = torch.randint(sample_len, (num_samples, ))
        else:
            self.target_key_inds = torch.empty((num_samples,), dtype=torch.long)
            for i in tqdm(range(num_samples)):
                unique_keys = self.keys[i].unique(dim=0)
                key = unique_keys[torch.randperm(len(unique_keys))[0]]
                try:
                    idx = torch.max(torch.where(torch.all(self.keys[i] == key, dim=-1))[0], dim=0)[0].long()
                except Exception:
                    print(f"{self.keys[i]}, {key}")
                    raise 1
                assert torch.all(self.keys[i][idx] == key)
                self.target_key_inds[i] = idx
    def __getitem__(self, idx):
        keys, values, tgt_ind = self.keys[idx], self.values[idx], self.target_key_inds[idx]
        # dim = 0 if keys.ndim == 1 else 1
        # keys = torch.chunk(keys, self.sample_len, dim=dim)
        # values = torch.chunk(values, self.sample_len, dim=dim)
        sample = {'keys': keys, 'values': values, 'target_key_ind': tgt_ind}
        return sample
    def __len__(self):
        return self.keys.shape[0]

def make_collator(args):
    sep_token, gen_token, eos_token = 100, 101, 102
    def collate_fn(batch, valid=False):
        keys = [b['keys'] for b in batch]
        values = [b['values'] for b in batch]
    
        if not args.vary_n_segments:
            tgt_inds = [b['target_key_ind'].item() for b in batch]
            n = len(keys[0])
        else:
            n = torch.randint(1, len(keys[0])+1, size=())
            keys = [x[-n:] for x in keys]
            values = [x[-n:] for x in values]
            if not rewrite_setting:
                tgt_inds = [torch.randint(0, n, size=()).item() for _ in range(len(keys))]
            else:
                tgt_inds = []
                for i in range(len(keys)):
                    unique_keys = keys[i].unique(dim=0)
                    key = unique_keys[torch.randperm(len(unique_keys))[0]]
                    try:
                        idx = torch.max(torch.where(torch.all(keys[i] == key, dim=-1))[0], dim=0)[0].long()
                    except Exception:
                        print(f"{keys[i]}, {key}")
                        raise 1
                    assert torch.all(keys[i][idx] == key)
                    tgt_inds.append(idx)



        bs = len(keys)
        sep_tokens = torch.ones(bs, 1) * sep_token
        eos_tokens = torch.ones(bs, 1) * eos_token
        gen_tokens = torch.ones(bs, 1) * gen_token
        sample = []

        for i in range(n):
            sample.append(torch.stack([k[i] for k in keys]))
            sample.append(sep_tokens)
            sample.append(torch.stack([v[i] for v in values]))
            sample.append(eos_tokens)

        target_keys = torch.stack([k[i] for i, k in zip(tgt_inds, keys)])
        target_values = torch.stack([k[i] for i, k in zip(tgt_inds, values)])

        sample.append(target_keys)
        sample.append(gen_tokens)

        input_ids_generate = torch.cat(sample, dim=1)

        sample.append(target_values)
        sample.append(eos_tokens)
        input_ids = torch.cat(sample, dim=1)

        labels_mask = torch.zeros_like(input_ids).bool()
        labels_mask[:, -args.value_size - 2:] = True

        collated = {'input_ids': input_ids.long(), 
                    'input_ids_generate': input_ids_generate.long(), 
                    'attention_mask': torch.ones_like(input_ids).bool(),
                    'attention_mask_generate': torch.ones_like(input_ids_generate).bool(),
                    'labels': input_ids.long(), 
                    'labels_mask': labels_mask, 
                    }
        return collated
    return collate_fn
