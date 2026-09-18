"""Create query-bound RESM features, either from RESM or existing .npy outputs."""
import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
from inference import read_query, load_features


def load_resm_model(source):
    """Adapt two training-only constructs in the pinned upstream release.

    Its dataset annotation refers to an absent class, and its unused training
    Config has mutable dataclass defaults incompatible with Python 3.11+.
    No network/model-forward AST nodes are changed; source files stay untouched.
    """
    path = source.resolve() / 'model.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == 'dataset':
            if [name.name for name in node.names] != ['TRRosettaContactDataset']:
                raise ValueError('Unexpected RESM dataset import; use the documented commit')
            tree.body[i] = ast.Assign(targets=[ast.Name(id='TRRosettaContactDataset', ctx=ast.Store())],
                                      value=ast.Name(id='object', ctx=ast.Load()))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ClassDef) and node.name == 'Config')]
    module = ModuleType('_evoribo_resm_dependency')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module.ESM2


def extract(query, source, weights, device):
    # RESM is a separate public dependency, not EvoRibo's private model source.
    sys.path.insert(0, str(source.resolve()))
    import torch
    import rna_esm
    ESM2 = load_resm_model(source)
    from evo.tokenization import Vocab
    alphabet = rna_esm.data.Alphabet(
        standard_toks=rna_esm.data.proteinseq_toks['toks'],
        prepend_toks=('<cls>', '<pad>', '<eos>', '<unk>'),
        append_toks=('<null_1>', '<mask>'), prepend_bos=True,
        append_eos=True, use_msa=False)
    vocab = Vocab.from_esm_alphabet(alphabet)
    config = SimpleNamespace(embed_dim=1280, num_attention_heads=20, num_layers=33)
    model = ESM2(vocab=vocab, model_config=config, token_dropout=False)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    for key in ('state_dict', 'model_state_dict', 'model'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if state and all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    mapping = dict(A='K', U='D', C='N', G='Y')
    tokens = [vocab.bos_idx] + [vocab.index(mapping[x]) for x in query] + [vocab.eos_idx]
    with torch.inference_mode():
        result = model(torch.tensor([tokens], device=device), repr_layers=[33], need_head_weights=True)
    length = len(query)
    embedding = result['representations'][33][0, 1:-1].float().cpu().numpy()
    attention = result['attentions'][0, :, :, 1:-1, 1:-1].reshape(660, length, length)
    return embedding, attention.float().cpu().numpy().transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--embedding', type=Path)
    parser.add_argument('--attention', type=Path)
    parser.add_argument('--resm-source', type=Path)
    parser.add_argument('--resm-weights', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    query = read_query(args.query)
    if args.out.exists():
        parser.error('Choose a new output path')
    if args.embedding and args.attention and not (args.resm_source or args.resm_weights):
        embedding = np.load(args.embedding, allow_pickle=False)
        attention = np.load(args.attention, allow_pickle=False)
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding[0]
        if attention.shape == (660, len(query), len(query)):
            attention = attention.transpose(1, 2, 0)
    elif args.resm_source and args.resm_weights and not (args.embedding or args.attention):
        embedding, attention = extract(query, args.resm_source, args.resm_weights, args.device)
    else:
        parser.error('Supply either embedding+attention or resm-source+resm-weights')
    if embedding.shape != (len(query), 1280) or attention.shape != (len(query), len(query), 660):
        parser.error('Incorrect RESM-650M shapes')
    if not np.isfinite(embedding).all() or not np.isfinite(attention).all():
        parser.error('Nonfinite RESM features')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as stream:
        np.savez_compressed(stream, query=query, embedding=embedding.astype(np.float32),
                            attention=attention.astype(np.float32))
    load_features(args.out, query)
    print(args.out)


if __name__ == '__main__':
    main()
