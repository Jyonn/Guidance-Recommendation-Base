"""Build movie-title features from Word2Vec-lab's exported vectors.txt."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from torch.nn import functional as F

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def build(vectors_path, metadata_path, output):
    titles = {}
    with metadata_path.open(encoding='latin-1') as stream:
        for line in stream:
            fields = line.rstrip('\n').split('|')
            item = int(fields[0])
            if item in titles:
                raise ValueError(f'Duplicate movie ID {item}')
            titles[item] = TOKEN_RE.findall(fields[1].lower())
    needed = {word for words in titles.values() for word in words}
    vectors = {}
    with vectors_path.open(encoding='utf-8') as stream:
        count, dim = map(int, stream.readline().split())
        if count < 1 or dim < 1:
            raise ValueError('Invalid Word2Vec header')
        actual = 0
        for line in stream:
            fields = line.split()
            if len(fields) != dim + 1:
                raise ValueError(f'Invalid vector at line {actual + 2}')
            actual += 1
            if fields[0] in needed:
                value = torch.tensor([float(v) for v in fields[1:]])
                if not torch.isfinite(value).all():
                    raise ValueError('Non-finite word vector')
                vectors[fields[0]] = value
        if actual != count:
            raise ValueError('Word2Vec header count does not match rows')
    ids = sorted(titles)
    features = torch.zeros(len(ids), dim)
    covered = 0
    matched_tokens = 0
    for index, item in enumerate(ids):
        found = [vectors[w] for w in titles[item] if w in vectors]
        matched_tokens += len(found)
        if found:
            covered += 1
            features[index] = F.normalize(torch.stack(found).mean(0), dim=0)
    if not covered:
        raise ValueError('No title tokens matched the exported vocabulary')
    report = dict(items=len(ids), covered_items=covered, zero_oov_items=len(ids)-covered,
                  matched_title_tokens=matched_tokens, title_tokens=sum(map(len, titles.values())),
                  unique_matched_tokens=len(vectors), dimension=dim,
                  tokenizer="lowercase [a-z]+(?:'[a-z]+)? (Word2Vec-lab English rule)",
                  pooling='mean of matched raw word vectors, then L2 normalization; OOV ignored',
                  vectors_sha256=hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
                  metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(item_ids=ids, features=features, report=report), output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def load_features(path, item_ids):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    values = saved['features']
    ids = saved['item_ids']
    if (values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] < 1
            or len(set(ids)) != len(ids) or not torch.isfinite(values).all()):
        raise ValueError('Invalid item feature matrix or IDs')
    mapping = {item: index for index, item in enumerate(ids)}
    if any(item not in mapping for item in item_ids):
        raise ValueError('Feature file is missing training item IDs; rebuild with complete u.item')
    return values[[mapping[item] for item in item_ids]].float()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vectors', type=Path, required=True, help='Word2Vec text export with V D header')
    parser.add_argument('--metadata', type=Path, required=True, help='MovieLens u.item (Latin-1)')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data/features/movie_titles.pt')
    args = parser.parse_args()
    print(json.dumps(build(args.vectors, args.metadata, args.output), indent=2))
