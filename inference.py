"""Local EvoRibo inference. No server and no original homolog MSA required."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ALPHABET = '-ACGU'


def read_query(path):
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            records.append('')
        elif records:
            records[-1] += line.upper().replace('T', 'U')
        else:
            raise ValueError('Query must be FASTA')
    if len(records) != 1 or not records[0] or set(records[0]) - set('ACGU'):
        raise ValueError('Provide one ungapped A/C/G/U query record')
    if not 2 <= len(records[0]) <= 200:
        raise ValueError('This release supports query length 2..200')
    return records[0]


def read_bpp(path, length):
    """Read SPOT-RNA dense .prob or one-based i,j,p triples."""
    matrix = np.loadtxt(path, comments='#', ndmin=2)
    if matrix.shape != (length, length):
        if matrix.shape[1] != 3:
            raise ValueError('BPP must be LxL or one-based i,j,probability triples')
        triples = matrix
        indices = triples[:, :2]
        if not np.isfinite(triples).all() or not np.equal(indices, np.floor(indices)).all():
            raise ValueError('Invalid BPP indices')
        if (indices < 1).any() or (indices > length).any():
            raise ValueError('BPP index outside query')
        matrix = np.zeros((length, length))
        for i, j, prob in triples:
            matrix[int(i)-1, int(j)-1] = prob
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError('BPP probabilities must be finite and in [0,1]')
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0)
    return matrix.astype(np.float32)


def load_features(path, query):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['query'].item()) != query:
            raise ValueError('RESM features were not bound to this exact query')
        embedding = archive['embedding'].astype(np.float32)
        attention = archive['attention'].astype(np.float32)
    length = len(query)
    if embedding.shape != (length, 1280) or attention.shape != (length, length, 660):
        raise ValueError('Expected RESM-650M [L,1280] and [L,L,660]')
    if not np.isfinite(embedding).all() or not np.isfinite(attention).all():
        raise ValueError('Nonfinite RESM features')
    return embedding, attention


def predict(model_path, query, bpp, embedding, attention, device, threads):
    import onnxruntime as ort
    providers = ['CPUExecutionProvider']
    if device == 'cuda':
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('Install onnxruntime-gpu for CUDA, or use --device cpu')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if device == 'cuda' and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('ONNX CUDA provider failed to initialize; refusing silent CPU fallback')
    inputs = dict(embedding=embedding[None], attention=attention[None], bpp=bpp[None],
                  tokens=np.array([['AUCG'.index(x) for x in query]], dtype=np.int64),
                  mask=np.ones((1, len(query)), dtype=bool))
    outputs = session.run(['scalar', 'fi', 'fij'], inputs)
    if any(not np.isfinite(x).all() for x in outputs):
        raise ValueError('Nonfinite model prediction')
    return outputs[0][0, 0], outputs[1][0], outputs[2][0]


def legalize(fi, fij, tolerance=1e-4):
    """IPF matches each joint distribution to the separately predicted marginals."""
    fi = np.asarray(fi, dtype=np.float64)
    if fi.ndim != 2 or fi.shape[1] != 5:
        raise ValueError('fi must be Lx5')
    length = len(fi)
    raw = np.asarray(fij, dtype=np.float64)
    if raw.shape != (length, length, 25):
        raise ValueError('ONNX fij output must be [L,L,25], channel=5*a+b')
    p = raw.reshape(length, length, 5, 5).copy()
    if not np.isfinite(fi).all() or not np.isfinite(p).all() or (fi < 0).any() or (p < 0).any():
        raise ValueError('Invalid predicted probabilities')
    fi = np.maximum(fi, 1e-8)
    fi /= fi.sum(-1, keepdims=True)
    p = np.maximum((p + p.transpose(1, 0, 3, 2)) / 2, 1e-8)
    p /= p.sum((-1, -2), keepdims=True)
    for iteration in range(256):
        p *= fi[:, None, :, None] / p.sum(-1, keepdims=True)
        p *= fi[None, :, None, :] / p.sum(-2, keepdims=True)
        p = (p + p.transpose(1, 0, 3, 2)) / 2
        error = max(float(np.max(abs(p.sum(-1) - fi[:, None, :]))),
                    float(np.max(abs(p.sum(-2) - fi[None, :, :]))))
        if error <= tolerance:
            break
    else:
        raise ValueError(f'Marginal projection failed: error={error}')
    index = np.arange(length)
    p[index, index] = np.eye(5)[None] * fi[:, :, None]
    return fi.astype(np.float32), p.transpose(0, 2, 1, 3).astype(np.float32), {
        'ipf_iterations': iteration+1, 'marginal_max_abs_error': error}


def sample_msa(fi, fij, query, args):
    import torch
    from adabmDCA.sampling import get_sampler
    from adabmDCA.training import train_graph
    from adabmDCA.utils import init_parameters, init_chains
    from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch has no CUDA device')
    fi = torch.tensor(fi, device=device)
    fij = torch.tensor(fij, device=device)
    length = len(query)
    params = init_parameters(fi=fi)
    chains = init_chains(num_chains=args.chains, L=length, q=5, fi=fi, device=device, dtype=torch.float32)
    mask = torch.ones((length, 5, length, 5), device=device, dtype=torch.bool)
    idx = torch.arange(length, device=device)
    mask[idx, :, idx, :] = False
    sampler = torch.jit.script(get_sampler('gibbs'))
    chains, params, _, history = train_graph(
        sampler=sampler, chains=chains, mask=mask, fi_target=fi, fij_target=fij,
        params=params, nsweeps=args.sweeps, lr=0.01, max_epochs=args.max_epochs,
        target_pearson=args.target_pearson,
        log_weights=torch.zeros(args.chains, device=device), progress_bar=True, l2_reg=0.0)
    with torch.no_grad():
        chosen = torch.randperm(args.chains, device=device)[:args.depth]
        samples = sampler(chains=chains[chosen].clone(), params=params, nsweeps=args.sample_sweeps)
        pi = get_freq_single_point(data=samples)
        pij = get_freq_two_points(data=samples)
        pearson, slope = get_correlation_two_points(fij=fij, pij=pij, fi=fi, pi=pi)
    rows = [''.join(ALPHABET[i] for i in row) for row in samples.argmax(-1).cpu().tolist()]
    metrics = {'sample_covariance_pearson': float(pearson), 'sample_covariance_slope': float(slope),
               'fit_epochs': len(history.get('Epochs', []))}
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError('Nonfinite fit metrics; inspect convergence before using this MSA')
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--bpp', type=Path, required=True, help='User-supplied SPOT-RNA .prob')
    parser.add_argument('--features', type=Path, required=True, help='Query-bound RESM feature .npz')
    parser.add_argument('--model', type=Path, required=True, help='Downloaded EvoRibo ONNX artifact')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--depth', type=int, default=1000, help='Synthetic rows, excluding query')
    parser.add_argument('--chains', type=int, default=10000)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--sweeps', type=int, default=5)
    parser.add_argument('--sample-sweeps', type=int, default=50)
    parser.add_argument('--target-pearson', type=float, default=0.95)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if min(args.depth, args.chains, args.max_epochs, args.sweeps, args.sample_sweeps, args.threads) < 1:
        parser.error('Counts must be positive')
    if args.depth > args.chains or not 0 < args.target_pearson < 1:
        parser.error('depth <= chains and 0 < target-pearson < 1 required')
    if args.out.exists():
        parser.error('Output directory exists; choose a new directory')
    query = read_query(args.query)
    bpp = read_bpp(args.bpp, len(query))
    embedding, attention = load_features(args.features, query)
    scalar, fi, fij = predict(args.model, query, bpp, embedding, attention, args.device, args.threads)
    fi, fij, projection = legalize(fi, fij)
    rows, metrics = sample_msa(fi, fij, query, args)
    args.out.mkdir(parents=True)
    text = '>query\n' + query + '\n'
    text += ''.join(f'>synthetic_{i+1}\n{row}\n' for i, row in enumerate(rows))
    (args.out / 'generated.a3m').write_text(text)
    (args.out / 'query.fasta').write_text('>query\n' + query + '\n')
    np.savez_compressed(args.out / 'predicted_stats.npz', scalar=scalar, fi=fi, fij=fij, alphabet=ALPHABET)
    metrics.update(projection, synthetic_depth=len(rows), total_depth=len(rows)+1,
                   query_sha256=hashlib.sha256(query.encode()).hexdigest(), seed=args.seed)
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(f'Wrote {len(rows)+1} aligned sequences to {args.out / "generated.a3m"}')


if __name__ == '__main__':
    main()
