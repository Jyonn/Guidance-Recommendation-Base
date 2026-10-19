import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluate_ranking import evaluate


class FixedModel:
    def load_state_dict(self, state):
        pass

    def eval(self):
        pass

    def __call__(self, users, items):
        # Item 0 is the highest-scoring history item and must be masked.
        return torch.tensor([100., 10., 9., 8., 7., 6., 5.])


class RankingTests(unittest.TestCase):
    def test_history_mask_macro_average_and_cutoffs(self):
        saved = dict(config=dict(model='mf', dim=1), user_ids=[1, 2],
                     item_ids=list(range(7)), model_state={})
        data = dict(train=[(0, 0, 1)], validation=[(0, 1, 0)],
                    test=[(0, 2, 1), (0, 6, 1), (1, 3, 0)])
        with patch('evaluate_ranking.torch.load', return_value=saved), \
             patch('evaluate_ranking.load_data', return_value=(data, saved['user_ids'], saved['item_ids'])), \
             patch('evaluate_ranking.build_model', return_value=FixedModel()):
            result = evaluate(Path('unused'), Path('unused'))
        self.assertEqual(result['evaluated_users'], 1)
        self.assertEqual(result['skipped_users'], 1)
        expected = (1 + 1 / math.log2(6)) / (1 + 1 / math.log2(3))
        for k in (5, 10, 20):
            self.assertEqual(result[f'Recall@{k}'], 1)
            self.assertAlmostEqual(result[f'NDCG@{k}'], expected)


if __name__ == '__main__':
    unittest.main()
