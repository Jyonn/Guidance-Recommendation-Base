"""Full-catalog test Recall/NDCG for saved best checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from train import MF, NCF, load_data


@torch.no_grad()
def evaluate(checkpoint, data_dir):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    data, users, items = load_data(data_dir)
    if users != saved['user_ids'] or items != saved['item_ids']:
        raise ValueError('Checkpoint ID mappings do not match evaluation data')
    model = (MF(len(users), len(items), config['dim']) if config['model'] == 'mf' else
             NCF(len(users), len(items), config['dim'], config['hidden'], config['dropout']))
    model.load_state_dict(saved['model_state'])
    model.eval()
    seen = {u: set() for u in range(len(users))}
    relevant = {u: set() for u in range(len(users))}
    for split in ('train', 'validation'):
        for u, i, _ in data[split]:
            seen[u].add(i)
    for u, i, y in data['test']:
        if y:
            relevant[u].add(i)
    totals = {f'{metric}@{k}': 0.0 for metric in ('Recall', 'NDCG') for k in (5, 10, 20)}
    evaluated = 0
    all_items = torch.arange(len(items))
    for u, positives in relevant.items():
        if not positives:
            continue
        if positives & seen[u]:
            raise ValueError('Test positive overlaps an excluded history item')
        scores = model(torch.full_like(all_items, u), all_items)
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite model scores')
        scores[list(seen[u])] = -torch.inf
        # Stable sorting makes equal scores resolve by ascending original item ID.
        count = min(20, len(items) - len(seen[u]))
        ranked = torch.argsort(scores, descending=True, stable=True)[:count].tolist()
        for k in (5, 10, 20):
            hits = [int(item in positives) for item in ranked[:k]]
            totals[f'Recall@{k}'] += sum(hits) / len(positives)
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(positives))))
            totals[f'NDCG@{k}'] += dcg / ideal
        evaluated += 1
    if not evaluated:
        raise ValueError('No users with test positives')
    return dict(evaluated_users=evaluated, skipped_users=len(users) - evaluated,
                **{key: value / evaluated for key, value in totals.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data/processed')
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for checkpoint in sorted(args.run_dir.glob('*/best.pt')):
        metrics = evaluate(checkpoint, args.data_dir)
        result = dict(experiment=checkpoint.parent.name, **metrics)
        (checkpoint.parent / 'ranking_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        print(json.dumps(result), flush=True)
    if not results:
        raise ValueError('No best.pt checkpoints found')
    protocol = dict(split='test', relevance='score > 3', candidates='all items in filtered training catalog',
                    exclude='all train and validation interactions, including explicit negatives',
                    aggregation='macro average over users with at least one test positive',
                    tie_break='ascending original item_id', results=results)
    (args.run_dir / 'ranking_summary.json').write_text(json.dumps(protocol, indent=2) + '\n')
    with (args.run_dir / 'ranking_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
