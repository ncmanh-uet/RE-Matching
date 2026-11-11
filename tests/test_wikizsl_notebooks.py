"""Validate notebook wiring offline; synthetic data/trainer do not measure accuracy."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WikiNotebookTest(unittest.TestCase):
    def test_three_notebooks(self):
        (ROOT / 'outputs').mkdir(exist_ok=True)
        for m in [5, 10, 15]:
            with self.subTest(m=m), tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as tmp:
                base = Path(tmp)
                notebook = json.loads((ROOT / f'notebooks/kaggle_wikizsl_m{m}.ipynb').read_text(encoding='utf-8'))
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), cell['id'], 'exec')
                namespace = {}
                exec(''.join(notebook['cells'][1]['source']), namespace)
                self.assertEqual(namespace['M_VALUES'], [m])
                self.assertEqual(namespace['SPLITS'], ['40', '41', '42', '43', '44'])
                self.assertEqual(namespace['OUTPUT_DIR'].name, f'wikizsl_m{m}')
                self.assertEqual([namespace[k] for k in ['BATCH_SIZE', 'EPOCHS', 'TRAIN_SEED', 'LR']],
                                 [128, 5, 42, 2e-6])
                relations = set()
                for split in namespace['SPLITS']:
                    partition = json.loads((ROOT / f'data/rel2id/wikizsl_rel2id/wikizsl_rel2id_{m}_{split}.json').read_text())
                    for labels in partition.values():
                        relations.update(labels)
                self.assertEqual(len(relations), 113)
                source = base / 'wikizsl_dataset.json'
                source.write_text(json.dumps([dict(token=['alice', 'met', 'bob'], relation=r,
                    h={'pos': [0, 1]}, t={'pos': [2, 3]}) for r in sorted(relations)]))
                namespace.update(REPO_DIR=ROOT, DATASET_FILE=str(source))
                preparation = ''.join(notebook['cells'][5]['source']).replace(
                    'Path("/kaggle/working/rematching-data")', f'Path({str(base / "data")!r})')
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(preparation, namespace)
                self.assertEqual(len(namespace['SPLIT_INFO']), 5)
                self.assertTrue(all(row['train_relations'] == 108 - m for row in namespace['SPLIT_INFO']))
                # An actual child process verifies command args, checkpoint matching,
                # run skipping, smoke isolation, and the m=10 CSV aggregation path.
                repo = base / 'repo'
                (repo / 'model').mkdir(parents=True)
                (repo / 'model/train.py').write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
for key in ['dataset','unseen','rel_split_seed','ckpt_dir']:
    p.add_argument('--'+key,required=True)
a,_=p.parse_known_args()
assert a.dataset=='wikizsl' and a.rel_split_seed in ['40','41','42','43','44']
d=Path(a.ckpt_dir); d.mkdir(parents=True,exist_ok=True)
f=d/f'wikizsl_split_{a.rel_split_seed}_unseen_{a.unseen}_f1_0.81234567'
f.write_text('synthetic checkpoint'); Path(str(f)+'.json').write_text('{}')
print('[test] final precision: 0.8000, recall: 0.8200, f1 score: 0.8123')
''')
                namespace.update(REPO_DIR=repo, OUTPUT_DIR=base / 'out', SIGNATURE='synthetic-test',
                                 TRAIN_CONFIG={'batch_size': 128, 'epochs': 5, 'seed': 42})
                exec(''.join(notebook['cells'][8]['source']), namespace)
                with contextlib.redirect_stdout(io.StringIO()):
                    for split in namespace['SPLITS']:
                        namespace['run_one'](m, split)
                    first = namespace['run_one'](m, '40')
                    self.assertEqual(first['status'], 'completed')
                    namespace['run_one'](m, '40', smoke=True)
                    summary = namespace['export_results']()
                self.assertEqual(len(summary), 1)
                self.assertEqual(summary[0]['m'], m)
                self.assertEqual(summary[0]['completed_runs'], 5)
                self.assertTrue(summary[0]['complete'])
                self.assertAlmostEqual(summary[0]['f1_mean_pct'], 81.234567)
                self.assertEqual(len((base / 'out/results.csv').read_text().splitlines()), 6)


if __name__ == '__main__':
    unittest.main()
