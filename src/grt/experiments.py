"""Named experiment composition; the final typed Config remains the run contract."""
from pathlib import Path
import yaml
from grt.config import Config, build_config, merge_dicts, validate_config

CATALOG = Path(__file__).resolve().parents[2] / 'configs' / 'experiments.yaml'


def parse_overrides(items):
    """Parse --set section.field=value using YAML scalar/list syntax."""
    result = {}
    for item in items:
        key, separator, value = item.partition('=')
        parts = key.split('.')
        if not separator or len(parts) < 2 or any(not part for part in parts):
            raise ValueError(f'Expected section.field=value, got {item!r}')
        node = result
        for part in parts[:-1]:
            if part in node and not isinstance(node[part], dict):
                raise ValueError(f'Conflicting override: {key}')
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return result


def load_experiment(*, task, model='rmt', size=None, tokenization=None,
                    budget=None, runtime=None, steps=None, overrides=None,
                    output_dir=None, catalog_path=CATALOG):
    catalog = yaml.safe_load(Path(catalog_path).read_text())
    if catalog.get('version') != 1:
        raise ValueError('Unsupported experiment catalog version')
    if task == 'wikitext103':
        size = size or 'small'
        tokenization = tokenization or 'gpt2_bpe'
        budget = budget or 'full'
    elif tokenization is not None:
        raise ValueError('Tokenization presets apply only to WikiText')
    raw = catalog['defaults']
    selections = [('tasks', task), ('sizes', size), ('models', model),
                  ('tokenizers', tokenization), ('budgets', budget), ('runtimes', runtime)]
    for section, name in selections:
        if name is None:
            continue
        choices = catalog[section]
        if name not in choices:
            raise ValueError(f'Unknown {section} preset {name!r}; choose {list(choices)}')
        raw = merge_dicts(raw, choices[name])
    if steps is not None:
        raw = merge_dicts(raw, {'training': {'max_steps': steps, 'curriculum': []}})
    raw = merge_dicts(raw, overrides or {})
    cfg = build_config(Config, raw)
    if output_dir is not None:
        cfg.run.output_dir = str(output_dir)
    return validate_config(cfg, require_output=output_dir is not None)
