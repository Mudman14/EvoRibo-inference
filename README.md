# EvoRibo Local Inference

Generate a pseudo-MSA from an RNA query and its **user-provided SPOT-RNA BPP**.
Everything runs on your machine. No account, API key, server, homolog search,
original MSA, or NuFold installation is required.

This repository contains the inference client and example inputs. EvoRibo's
original model Python source and training code are not included. The trained
model is distributed separately as a versioned ONNX inference artifact.
ONNX contains an inspectable computation graph and weights; it is not encryption
or a guarantee against reverse engineering.

## Install

Use Python 3.12+ in a separate environment:

```bash
git clone https://github.com/Mudman14/EvoRibo-inference.git
cd EvoRibo-inference
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_model.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead.
The model download is approximately 493 MB and is SHA256-verified.

## Try an example

Two examples include query-bound, precomputed RESM-650M features and SPOT-RNA
BPP: **5NEF_A (18 nt)** and **1GAX_C (75 nt)**. They are format/inference
examples, not a claim of model quality on independent benchmark data.

Fast CPU pipeline smoke (32 synthetic rows, deliberately short Potts fit):

```bash
python inference.py --query examples/5NEF_A/query.fasta --bpp examples/5NEF_A/bpp.prob --features examples/5NEF_A/features.npz --model models/evoribo.onnx --out outputs/smoke --depth 32 --chains 128 --max-epochs 20 --sweeps 2 --sample-sweeps 5 --target-pearson 0.6
```

For the normal sampling budget, omit the smoke overrides:

```bash
python inference.py --query examples/1GAX_C/query.fasta --bpp examples/1GAX_C/bpp.prob --features examples/1GAX_C/features.npz --model models/evoribo.onnx --out outputs/1GAX_C
```

Default: query + 1,000 synthetic rows; 10,000 fitting chains; at most 2,000
Potts updates. CPU fitting can be slow. A finite update budget does not guarantee
convergence; inspect the reported covariance correlation before downstream use.

Outputs:

```text
generated.a3m       # Query first, then synthetic rows of the same length
query.fasta
predicted_stats.npz # Scalar map, legalized fi/fij, alphabet
metrics.json       # Projection error, fit/sample metrics, depth and seed
```

No homolog MSA is used. adabmDCA fits a per-query Potts model to the predicted
statistics and samples it; this is not simply decoding a stored training MSA.

## Your own RNA

1. Provide a single-record, ungapped FASTA query (A/C/G/U; T is converted to U).
   This release accepts 2-200 nt; this bound does not establish generalization
   to every RNA in that range.
2. Run [SPOT-RNA](https://github.com/jaswindersingh2/SPOT-RNA) separately and
   provide its `.prob` BPP for the same query. **BPP is mandatory.** No automatic
   ViennaRNA or all-zero secondary-structure substitution is performed.
3. Generate RESM-650M-KDNY features and bind them to the exact query below.
4. Run `inference.py` as in the examples, using your three input paths.

### Existing RESM outputs

```bash
python prepare_features.py --query query.fasta --embedding query_emb.npy --attention query_atp.npy --out features.npz
```

Accepted RESM shapes: embedding `[L,1280]` or `[1,L,1280]`; attention
`[660,L,L]` or `[L,L,660]`. BOS/EOS positions must already be removed. Binding
records your assertion that these arrays belong to the query; shape checks alone
cannot establish their biological identity.

### Extract RESM features from the query

[RESM](https://github.com/yikunpku/RESM) is a separate public model. Install its
dependencies in an appropriate environment, obtain its **RESM-650M-KDNY.pt**
weights from the authors' [checkpoint record](https://zenodo.org/records/15980876),
and clone its code. Do not use the 150M weights or protein ESM2 weights instead.
The adapter targets RESM commit `7ae3e4172d2eb8d29826d58b6d66535d16846618`.

```bash
python -m pip install -r requirements-resm.txt
git clone https://github.com/yikunpku/RESM.git
git -C RESM checkout 7ae3e4172d2eb8d29826d58b6d66535d16846618
python prepare_features.py --query query.fasta --resm-source RESM --resm-weights RESM-650M-KDNY.pt --device cuda --out features.npz
```

Feature extraction can run in the RESM environment; the resulting `.npz` is
portable to the inference environment. The adapter uses the authors' KDNY token
mapping, last-layer embeddings, all 33x20 attention channels, and removes
BOS/EOS. Only trusted tensor-only checkpoints are loaded (`weights_only=True`).
The adapter bypasses two training-only upstream import/configuration issues
in memory; it does not modify RESM source files or the network's forward pass.

## GPU inference

Install a CUDA-enabled PyTorch build appropriate for your GPU. Replace the CPU
ONNX runtime with the compatible GPU runtime (do not install both):

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Then add `--device cuda`. Missing CUDA support fails explicitly, rather than
silently running the whole calculation on CPU. Hardware-dependent provider
fallbacks for individual ONNX operators may still occur.

## Validation and limitations

Run `python -m unittest discover -s tests -v`.
`validation.json` records real-input ONNX/PyTorch numerical comparisons. Export
was also checked at odd/even lengths 16, 17 and 24. Tests cover input mismatch,
probability validity, exchange symmetry and marginal projection.

The scalar map is the trained head's normalized output, not an inferred absolute
MI scale. The ONNX pair output is `[1,L,L,25]`, channel `5*a+b`, using
alphabet `-ACGU`. The saved, legalized `predicted_stats.npz` uses `[L,5,L,5]`.
Iterative proportional fitting matches its marginals to predicted fi. This
projection does not guarantee global realizability or excellent Potts fitting.

The quick smoke budget checks execution and output format, not scientific
quality. RESM/SPOT-RNA installation and full-quality GPU runs require their own
validation in your environment. The supplied feature-bundle path was tested
end to end on CPU; the upstream RESM adapter was import-tested, but its full
650M weight-loading/extraction path and ONNX CUDA execution have not yet been
validated for this release. Do not substitute a different BPP predictor
without treating that change as an input-distribution ablation.

## Source boundaries

Only the local runner, statistics formatting, public dependency adapter, and
test fixtures are published here. No development repository history, original
training checkpoint, optimizer state, private model Python classes, local paths,
cluster credentials, or NuFold code are included.

RESM and adabmDCA retain their respective authorship and licenses. Please cite
the original RESM and adabmDCA work in addition to EvoRibo when using this pipeline.
