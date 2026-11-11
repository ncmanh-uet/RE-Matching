# Modern library support and Kaggle

## Tested environment

On 2026-11-10 the user's new installation resolved to Python 3.14.3,
PyTorch 2.15.0+cpu, Transformers 5.22.2, Sentence Transformers 6.2.0,
NumPy 2.5.4 and scikit-learn 1.9.1. `pip check` reported no conflicts.
`requirements.txt` pins the non-Torch direct dependencies to those versions;
`requirements-tested.txt` also pins the tested Torch release. This is a tested
snapshot, not an automatic promise of compatibility with future releases.

Kaggle has its own Python/CUDA stack. Check it first and retain its CUDA-enabled
Torch when it satisfies `torch>=2.6`. Do not install the local Windows CPU wheel
on Kaggle. A different Torch version still requires a GPU smoke test.

## Kaggle setup

Enable Internet and a GPU accelerator in the notebook settings. Clone your fork
and checkout the commit containing these changes. From the repository directory:

```python
import sys
import torch
print(sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
assert sys.version_info >= (3, 12)
assert torch.cuda.is_available(), 'Enable a GPU accelerator'
```

```python
%pip install -r requirements.txt
```

Restart the notebook session if already-imported packages were replaced. Run
`pip check` using the notebook's Python interpreter after installation.

Attach the paper's processed dataset as Kaggle input. The expected file is
`DATA_ROOT/fewrel/fewrel_dataset.json` or
`DATA_ROOT/wikizsl/wikizsl_dataset.json`. The code expects a JSON list of samples
with `token`, `relation`, `h.pos` and `t.pos`; positions are zero-based,
end-exclusive token spans. The repository already includes split definitions
and description files, but does not include the full datasets.

For example, stage FewRel in the writable working directory:

```python
!cp /kaggle/input/YOUR_DATASET/fewrel_dataset.json data/fewrel/fewrel_dataset.json
```

Run two optimizer steps and evaluation before launching the full experiment:

```python
!python model/train.py --device cuda --dataset fewrel --unseen 10 --rel_split_seed ori --batch_size 4 --eval_batch_size 16 --epochs 1 --max_steps 2 --ckpt_dir /kaggle/working/smoke-ckpt
```

This still loads/preprocesses the dataset and evaluates the full test split.
Then launch the paper's configuration:

```python
!python model/train.py --device cuda --dataset fewrel --unseen 10 --rel_split_seed ori --batch_size 128 --eval_batch_size 32 --epochs 5 --ckpt_dir /kaggle/working/ckpt
```

Batch size 128 may exceed available GPU memory. Start small to establish memory
usage. Changing training batch size changes in-batch negative sampling and
therefore the experiment; record it explicitly. Lowering `eval_batch_size` only
changes evaluation batching. The default visible device is GPU 0; opt into
multiple GPUs with `--visible_device 0,1`. CPU is supported with `--device cpu`.

Checkpoint filenames retain the original format. Each new checkpoint has a JSON
sidecar recording CLI arguments, direct library versions and CUDA/GPU details.
Save a Kaggle notebook version with outputs to retain checkpoints. To evaluate:

```python
!python model/inference.py --device cuda --ckpt_save_path /kaggle/working/ckpt --ckpt_name fewrel_split_ori_unseen_10_f1_SCORE
```

Replace `SCORE` with the actual filename's score. Pass the same `--alpha`,
`--gamma`, `--pretrained_model`, `--description_model`, `--max_length`, and
description/data files as training when overriding defaults. Paths default to
the repository rather than the current working directory. Internet is needed
for uncached Hugging Face models; `--pretrained_model` and `--description_model`
also accept local directories.

## Compatibility changes

- Initialize custom heads through `post_init()` for modern Hugging Face loading.
  The backbone is explicitly restricted to BERT, matching the actual model
  implementation; descriptions must have matching embedding dimensions.
- Create tensors on the input/model device instead of hardcoding `.cuda()`.
- Cap negative sampling at the local batch size, including a final batch of one.
  The original loss and negative-score floor are retained.
- Use actual minibatch size to normalize matching loss, one combined backward
  pass, and a scheduler based on `len(trainloader)`, including partial batches.
- Clamp zero projection denominators to avoid NaNs and reject non-finite loss.
- Resolve marker IDs from the tokenizer. Track inserted markers separately from
  literal `#`/`@` in the text, crop long context while preserving both entities,
  support nested, crossing and identical spans without duplicating source text, and fail clearly for invalid spans or entities too far apart to fit.
- Order relation vectors by numeric label ID and stack NumPy arrays before
  conversion to tensors. Evaluate embeddings on CPU to reduce GPU memory use.
- Create checkpoint directories, save unwrapped model weights, and load with
  `weights_only=True`/CPU mapping. Accept legacy `module.`-prefixed checkpoints.
- Keep the original paper's custom macro metric unchanged for comparability.

These changes can alter initialization, final partial-batch updates and treatment
of long inputs compared with the original environment. Reproduction scores must
be measured; they are not guaranteed to match the paper bit for bit.

## Offline verification

```shell
python -m unittest discover -s tests -v
```

Tests create a tiny local BERT and Sentence Transformer without downloading
models. They check actual pretrained-weight loading, finite gradients in both
heads and backbone, optimizer steps for batches of three and one with seven
requested negatives, marker/cropping behavior, legacy checkpoint loading, and
the real training/evaluation/checkpoint/inference CLIs over a synthetic dataset
for two epochs. No full-data training, real Hub checkpoint download or CUDA
execution has been verified locally. Passing these tests cannot rule out GPU
OOM, malformed unseen data or differences in the Kaggle environment.

References: [Transformers v5 migration](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md),
[PyTorch checkpoint conventions](https://docs.pytorch.org/tutorials/beginner/saving_loading_models/),
[SentenceTransformer API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).


## Wiki-ZSL overlapping entities (2026-11-11)

The processed dataset contains 828 overlapping entity pairs among 93,383 samples.
For example, sample 23 pairs `Virginia House of Delegates` with `Virginia`.
These valid samples are retained. `model/entity_spans.py` inserts markers at
word boundaries and independently tracks the lexical subwords belonging to each
entity. Shared lexical tokens belong to both entity masks; inserted markers
belong to neither. Disjoint spans retain the previous token stream.

The three Wiki-ZSL notebooks no longer reject overlap during preflight. They
check that the cloned code contains the updated span helper before training.
Push both `model/entity_spans.py` and `model/data_process.py` to GitHub, upload
the updated notebook, and start a fresh Kaggle session to clone the new HEAD.
An existing working clone is not pulled automatically.

Five span tests passed, including bookkeeping over all local Wiki-ZSL samples
using a word-tokenizer test double; notebook runner tests use synthetic data
and a fake trainer. The prior local venv is no longer present, so the updated
real BERT forward/backward regression test has not been run locally. Run the
notebook GPU smoke test to verify the installed tokenizer/model stack.
