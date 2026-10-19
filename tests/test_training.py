import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import MF, batch_loss, make_examples, parser, run


class TrainingTests(unittest.TestCase):
    def test_sampling_and_pairing(self):
        rows = [(0, 0, 1), (0, 1, 0), (1, 2, 1), (1, 3, 0)]
        sampled = make_examples(rows, 'bce', 'random', 5, 4, 7)
        observed = {0: {0, 1}, 1: {2, 3}}
        for u, i, y in sampled:
            if y == 0:
                self.assertNotIn(int(i), observed[int(u)])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            sampled.tensors, make_examples(rows, 'bce', 'random', 5, 4, 7).tensors)))
        pairs = make_examples(rows, 'bpr', 'real', 5, 1, 7)
        self.assertEqual([tuple(map(int, row)) for row in pairs], [(0, 0, 1), (1, 2, 3)])

    def test_bpr_direction(self):
        model = MF(1, 2, 1)
        with torch.no_grad():
            model.user.weight.fill_(1)
            model.item.weight.copy_(torch.tensor([[2.0], [-2.0]]))
        good = batch_loss(model, (torch.tensor([0]), torch.tensor([0]), torch.tensor([1])), 'bpr', 'cpu')
        bad = batch_loss(model, (torch.tensor([0]), torch.tensor([1]), torch.tensor([0])), 'bpr', 'cpu')
        self.assertLess(good.item(), bad.item())

    def test_all_combinations_and_early_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('train', 'validation', 'test'):
                with (root / f'{name}.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['user_id', 'item_id', 'score', 'timestamp', 'label'])
                    # Sparse original IDs; each user has an unseen training-catalog item.
                    writer.writerows([(10, 100, 5, 1, 1), (10, 200, 1, 2, 0),
                                      (20, 300, 4, 1, 1), (20, 200, 2, 2, 0)])
            for model in ('mf', 'ncf'):
                for mode in ('real', 'random'):
                    for loss in ('bce', 'mse', 'bpr'):
                        with self.subTest(model=model, mode=mode, loss=loss):
                            output = root / f'{model}-{mode}-{loss}'
                            args = parser().parse_args([
                                '--data-dir', str(root), '--output-dir', str(output),
                                '--model', model, '--neg-mode', mode, '--loss', loss,
                                '--epochs', '5', '--patience', '1', '--min-delta', '100',
                                '--dim', '4', '--hidden', '8', '--batch-size', '2'])
                            result = run(args)
                            self.assertEqual(result['best_epoch'], 1)
                            self.assertEqual(result['epochs_run'], 2)
                            self.assertTrue(result['stopped_early'])
                            # Identical validation/test fixtures prove best state is restored.
                            self.assertAlmostEqual(result['test_loss'], result['best_validation_loss'])
                            self.assertTrue((output / 'best.pt').is_file())
                            self.assertTrue((output / 'metrics.json').is_file())


if __name__ == '__main__':
    unittest.main()
