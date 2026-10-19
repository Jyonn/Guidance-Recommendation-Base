"""Minimal MF/NCF training with explicit or sampled negatives."""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent


class MF(nn.Module):
    def __init__(self, users, items, dim):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def forward(self, user, item):
        return (self.user(user) * self.item(item)).sum(-1)


class NCF(nn.Module):
    """Embedding concatenation followed by a plain MLP."""
    def __init__(self, users, items, dim, hidden, dropout):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        layers = []
        width = 2 * dim
        for size in hidden:
            layers.extend([nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)])
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user, item):
        return self.mlp(torch.cat([self.user(user), self.item(item)], -1)).squeeze(-1)


def load_data(directory):
    raw = {}
    for name in ('train', 'validation', 'test'):
        with (directory / f'{name}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'{name} is empty')
        raw[name] = [(int(r['user_id']), int(r['item_id']), int(r['label'])) for r in rows]
        if any(int(r['score']) == 3 or int(r['label']) != int(int(r['score']) > 3) for r in rows):
            raise ValueError('Expected preprocessed binary labels with score=3 removed')
    user_ids = sorted({u for u, _, _ in raw['train']})
    item_ids = sorted({i for _, i, _ in raw['train']})
    umap = {u: n for n, u in enumerate(user_ids)}
    imap = {i: n for n, i in enumerate(item_ids)}
    data = {name: [(umap[u], imap[i], y) for u, i, y in rows] for name, rows in raw.items()}
    return data, user_ids, item_ids


def make_examples(rows, loss, mode, n_items, negatives, seed):
    rng = random.Random(seed)
    positive, negative, observed = defaultdict(list), defaultdict(list), defaultdict(set)
    for u, i, y in rows:
        (positive if y else negative)[u].append(i)
        observed[u].add(i)
    if mode == 'random':
        # Training catalog only: unseen items union the user's real negatives.
        negative = {u: sorted((set(range(n_items)) - observed[u]) | set(negative[u]))
                    for u in positive}
    if loss == 'bpr':
        examples = [(u, p, rng.choice(negative[u]))
                    for u, items in positive.items() if negative.get(u)
                    for p in items for _ in range(negatives)]
        dtype = torch.long
    elif mode == 'real':
        examples = rows
        dtype = torch.float32
    else:
        examples = []
        for u, items in positive.items():
            for p in items:
                examples.append((u, p, 1))
                if negative.get(u):
                    examples.extend((u, rng.choice(negative[u]), 0) for _ in range(negatives))
        dtype = torch.float32
    if not examples:
        raise ValueError('No usable examples; BPR needs a user with both positive and negative items')
    u, i, target = zip(*examples)
    return TensorDataset(torch.tensor(u), torch.tensor(i), torch.tensor(target, dtype=dtype))


def batch_loss(model, batch, loss, device):
    u, i, target = (x.to(device) for x in batch)
    scores = model(u, i)
    if loss == 'bpr':
        return F.softplus(model(u, target) - scores).mean()
    if loss == 'bce':
        return F.binary_cross_entropy_with_logits(scores, target)
    return F.mse_loss(scores.sigmoid(), target)


@torch.no_grad()
def evaluate(model, dataset, args, device):
    model.eval()
    total = 0.0
    for batch in DataLoader(dataset, batch_size=args.batch_size):
        total += batch_loss(model, batch, args.loss, device).item() * len(batch[0])
    return total / len(dataset)


def run(args):
    if min(args.dim, args.batch_size, args.epochs, args.patience, args.negatives, args.threads) < 1:
        raise ValueError('Integer hyperparameters must be positive')
    if (not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.min_delta, args.dropout))
            or args.lr <= 0 or min(args.weight_decay, args.min_delta) < 0
            or not 0 <= args.dropout < 1 or any(h < 1 for h in args.hidden)):
        raise ValueError('Invalid LR, weight decay, min delta, dropout or hidden dimensions')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data, user_ids, item_ids = load_data(args.data_dir)
    model = (MF(len(user_ids), len(item_ids), args.dim) if args.model == 'mf' else
             NCF(len(user_ids), len(item_ids), args.dim, args.hidden, args.dropout)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Both training modes share fixed, real-label validation/test data.
    validation = make_examples(data['validation'], args.loss, 'real', len(item_ids), args.negatives, args.seed)
    test = make_examples(data['test'], args.loss, 'real', len(item_ids), args.negatives, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    best, best_epoch, stale = float('inf'), 0, 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        training = make_examples(data['train'], args.loss, args.neg_mode, len(item_ids), args.negatives, args.seed + epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator)
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = batch_loss(model, batch, args.loss, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch[0])
        val = evaluate(model, validation, args, device)
        record = dict(epoch=epoch, train_loss=total / len(training), validation_loss=val,
                      train_examples=len(training))
        history.append(record)
        print(json.dumps(record), flush=True)
        if val < best - args.min_delta:
            best, best_epoch, stale = val, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError('No finite validation checkpoint')
    model.load_state_dict(best_state)
    torch.save(dict(model_state=best_state, config=config, user_ids=user_ids, item_ids=item_ids,
                    best_epoch=best_epoch), args.output_dir / 'best.pt')
    result = dict(config=config, best_epoch=best_epoch, best_validation_loss=best,
                  test_loss=evaluate(model, test, args, device), epochs_run=len(history),
                  stopped_early=len(history) < args.epochs, history=history,
                  validation_examples=len(validation), test_examples=len(test))
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('config', 'history')}))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['mf', 'ncf'], default='mf')
    p.add_argument('--neg-mode', choices=['real', 'random'], default='real')
    p.add_argument('--loss', choices=['bce', 'mse', 'bpr'], default='bce')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'runs/default')
    p.add_argument('--dim', type=int, default=32)
    p.add_argument('--hidden', type=int, nargs='+', default=[64, 32])
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=0.0001)
    p.add_argument('--negatives', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', default='cpu')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
