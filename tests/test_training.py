"""Offline integration tests: no Hub download or external dataset required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, models
from data_process import FewRelDataset, create_mini_batch
from model import REMatchingModel, emb_proj
from runtime import load_checkpoint, ordered_labels


class TrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT / 'outputs').mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        cls.base = Path(cls.tmp.name)
        cls.backbone = cls.base / 'bert'
        cls.backbone.mkdir()
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '#', '@',
                 'alice', 'bob', 'met', 'the', 'entity', 'types', 'including', 'and', 'person']
        BertTokenizer(vocab={word: i for i, word in enumerate(vocab)}).save_pretrained(cls.backbone)
        cls.config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=64, num_labels=2)
        cls.config.margin, cls.config.alpha = .06, .33
        cls.bert = BertModel(cls.config)
        cls.bert.save_pretrained(cls.backbone)
        cls.description = cls.base / 'description'
        encoder = SentenceTransformer(modules=[models.Transformer(str(cls.backbone)), models.Pooling(16)])
        encoder.save(str(cls.description))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def args(self):
        return SimpleNamespace(pretrained_model=str(self.backbone), max_length=32)

    def sample(self, relation='r0'):
        return {'token': ['alice', 'met', 'bob'], 'h': {'pos': [0, 1]},
                't': {'pos': [2, 3]}, 'relation': relation}

    def test_backward_small_batches_and_checkpoint(self):
        model = REMatchingModel.from_pretrained(str(self.backbone), config=self.config)
        # Verify HF v5 actually loaded pretrained backbone weights.
        torch.testing.assert_close(model.bert.embeddings.word_embeddings.weight,
                                   self.bert.embeddings.word_embeddings.weight)
        vectors = {r: [np.random.default_rng(i).normal(size=16).astype('float32')] * 3
                   for i, r in enumerate(['r0', 'r1'])}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for size in [3, 1]:
            dataset = FewRelDataset(self.args(), 'train',
                                    [self.sample(f'r{i % 2}') for i in range(size)], vectors, {'r0': 0, 'r1': 1})
            for sample, (head, tail) in zip(dataset.data, [([0, 3], [0, 1]), ([0, 2], [1, 3]), ([0, 1], [0, 1])]):
                sample['h']['pos'], sample['t']['pos'] = head, tail
            data = create_mini_batch([dataset[i] for i in range(size)])
            outputs, *_ = model(input_ids=data[0], e1_mask=data[1], e2_mask=data[2],
                                marked_head=data[3], marked_tail=data[4], attention_mask=data[5],
                                input_relation_emb=data[6], input_relation_head_emb=data[7],
                                input_relation_tail_emb=data[8], labels=data[9], num_neg_sample=7)
            loss = outputs[0] / size + outputs[1]
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertIsNotNone(model.code.weight.grad)
            self.assertIsNotNone(model.bert.embeddings.word_embeddings.weight.grad)
            optimizer.step()
        checkpoint = self.base / 'legacy.pt'
        torch.save({'module.' + k: v for k, v in model.state_dict().items()}, checkpoint)
        restored = REMatchingModel(self.config)
        load_checkpoint(restored, checkpoint)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[key])

    def test_entity_window_and_literal_markers(self):
        sample = {'token': ['#'] + ['met'] * 40 + ['alice', '@', 'bob'] + ['met'] * 40,
                  'h': {'pos': [41, 42]}, 't': {'pos': [43, 44]}, 'relation': 'r0'}
        vectors = {'r0': [np.ones(16, dtype='float32')] * 3}
        dataset = FewRelDataset(self.args(), 'test', [sample], vectors, {'r0': 0})
        data = dataset[0]
        self.assertEqual(len(data[0]), 32)
        self.assertEqual(data[4].sum(), 1)
        self.assertEqual(data[5].sum(), 1)
        # Reverse entity order must preserve head/tail identity.
        sample['h'], sample['t'] = sample['t'], sample['h']
        data = dataset[0]
        self.assertEqual(data[0][data[4].bool()].item(), dataset.tokenizer.convert_tokens_to_ids('bob'))
        torch.testing.assert_close(emb_proj(torch.ones(2, 16), torch.zeros(2, 16)), torch.zeros(2, 16))
        self.assertEqual(ordered_labels({'b': 1, 'a': 0}), ['a', 'b'])

    def test_train_and_inference_cli(self):
        data = self.base / 'data'
        desc = data / 'fewrel' / 'relation_description'
        desc.mkdir(parents=True, exist_ok=True)
        split = data / 'rel2id' / 'fewrel_rel2id'
        split.mkdir(parents=True, exist_ok=True)
        (split / 'fewrel_rel2id_2_ori.json').write_text(json.dumps({
            'train': {'r1': 1, 'r0': 0}, 'test': {'r3': 1, 'r2': 0}}))
        (data / 'fewrel' / 'fewrel_dataset.json').write_text(json.dumps(
            [self.sample(r) for r in ['r0', 'r1', 'r0', 'r2', 'r3']]))
        (desc / 'relation_description_addsynonym_processed.json').write_text(json.dumps([
            {'relation': f'r{i}', 'description': ['alice', 'met', 'bob'], 'h': ['person'], 't': ['person']}
            for i in range(4)]))
        common = ['--device', 'cpu', '--pretrained_model', str(self.backbone),
                  '--description_model', str(self.description), '--dataset_path', str(data),
                  '--rel2id_path', str(data / 'rel2id'), '--max_length', '32']
        output = self.base / 'ckpt'
        def run(script, extra):
            result = subprocess.run([sys.executable, str(ROOT / 'model' / script), *common, *extra],
                                    cwd=self.base, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('final precision:', result.stdout)
        run('train.py', ['--unseen', '2', '--epochs', '2', '--batch_size', '2', '--ckpt_dir', str(output)])
        checkpoint = next(path for path in output.iterdir() if path.suffix != '.json')
        self.assertTrue(Path(str(checkpoint) + '.json').exists())
        run('inference.py', ['--ckpt_save_path', str(output), '--ckpt_name', checkpoint.name])
        smoke_output = self.base / 'smoke-ckpt'
        run('train.py', ['--unseen', '2', '--epochs', '5', '--batch_size', '2',
                         '--max_steps', '1', '--ckpt_dir', str(smoke_output)])
        self.assertTrue(any(smoke_output.iterdir()))


if __name__ == '__main__':
    unittest.main()
