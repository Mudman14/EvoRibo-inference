# EvoRibo Inference

Generate RNA pseudo-MSAs locally from a query sequence, RESM features, and **SPOT-RNA BPP**.

## Installation

Use Python 3.12+:

```bash
git clone https://github.com/Mudman14/EvoRibo-inference.git
cd EvoRibo-inference
python -m pip install -r requirements.txt
python download_model.py
```

## Run an example

Example inputs are provided in `examples/5NEF_A` and `examples/1GAX_C`.

```bash
python inference.py --query examples/5NEF_A/query.fasta --bpp examples/5NEF_A/bpp.prob --features examples/5NEF_A/features.npz --model models/evoribo.onnx --out outputs/5NEF_A
```

The output `generated.a3m` contains the query followed by **1,000 generated sequences**. Change this with `--depth`. Predicted statistics and sampling metrics are also saved.

For a quick execution test, append `--depth 32 --chains 128 --max-epochs 20 --sweeps 2 --sample-sweeps 5`. This short run is not intended for production-quality MSAs.

## Use your own sequence

Provide a single-query FASTA (2-200 nt), its [SPOT-RNA](https://github.com/jaswindersingh2/SPOT-RNA) `.prob` file, and RESM-650M-KDNY embedding/attention outputs for the same sequence.

Package existing RESM outputs:

```bash
python prepare_features.py --query query.fasta --embedding query_emb.npy --attention query_atp.npy --out features.npz
python inference.py --query query.fasta --bpp query.prob --features features.npz --model models/evoribo.onnx --out outputs/my_query
```

To compute new RESM features, obtain `RESM-650M-KDNY.pt` from the [RESM authors](https://zenodo.org/records/15980876), then run:

```bash
python -m pip install -r requirements-resm.txt
git clone https://github.com/yikunpku/RESM.git
git -C RESM checkout 7ae3e4172d2eb8d29826d58b6d66535d16846618
python prepare_features.py --query query.fasta --resm-source RESM --resm-weights RESM-650M-KDNY.pt --device cuda --out features.npz
```

## GPU

With CUDA-enabled PyTorch installed:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Add `--device cuda` to the inference command. CPU execution is the default.

**Preview status:** CPU inference with the supplied features has been tested. Full RESM weight extraction and ONNX GPU execution remain unverified. Check sampling metrics before downstream use.
