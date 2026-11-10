"""Shared runtime helpers for local and Kaggle execution."""
import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Enable a Kaggle GPU accelerator.')
    return device


def ordered_labels(mapping):
    if sorted(int(i) for i in mapping.values()) != list(range(len(mapping))):
        raise ValueError('Relation IDs must be contiguous and start at zero.')
    return sorted(mapping, key=lambda key: int(mapping[key]))


def validate_data(args, training_data, test_data, train_desc, test_desc, train_labels, test_labels):
    if not training_data or not test_data:
        raise ValueError('Both training and test splits must contain samples.')
    for labels, descriptions in [(train_labels, train_desc), (test_labels, test_desc)]:
        if set(labels) != {item['relation'] for item in descriptions}:
            raise ValueError('Missing relation descriptions for this split.')


def save_run(args, model, checkpoint):
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(core.state_dict(), checkpoint)
    metadata = dict(vars(args))
    metadata['versions'] = {name: version(name) for name in
                            ['torch', 'transformers', 'sentence-transformers', 'numpy', 'scikit-learn']}
    metadata['cuda'] = torch.version.cuda
    metadata['gpu'] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    checkpoint.with_name(checkpoint.name + '.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')


def load_checkpoint(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    # Accept legacy checkpoints saved by DataParallel too.
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state)
