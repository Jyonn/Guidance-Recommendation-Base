import csv
import tempfile
import unittest
from pathlib import Path

import torch

from item_features import build, load_features
from train import build_model, parser, run
from evaluate_ranking import evaluate


class ItemFeatureTests(unittest.TestCase):
    def test_pooling_alignment_oov_and_frozen_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'vectors.txt').write_text('2 2\ntoy 1 0\nstory 0 1\n')
            (root/'u.item').write_text('20|Unknown (1995)|\n10|Toy Story (1995)|\n')
            path = root/'features.pt'
            report = build(root/'vectors.txt', root/'u.item', path)
            self.assertEqual(report['zero_oov_items'], 1)
            features = load_features(path, [20, 10])
            self.assertTrue(torch.equal(features[0], torch.zeros(2)))
            self.assertTrue(torch.allclose(features[1], torch.tensor([2**-.5, 2**-.5])))
            with self.assertRaises(ValueError):
                load_features(path, [999])
            for kind in ('mf', 'ncf'):
                model = build_model(dict(model=kind, dim=4, hidden=[8], dropout=0), 2, 2, features)
                model(torch.tensor([0, 1]), torch.tensor([0, 1])).sum().backward()
                self.assertIsNone(model.item_features.weight.grad)
                self.assertIsNotNone(model.feature_projection.weight.grad)
                self.assertTrue(torch.equal(model.feature_projection(features[0]), torch.zeros(4)))

    def test_all_feature_training_modes_and_portable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = dict(train=[(10,100,5,1,1),(10,200,1,2,0),(20,300,4,1,1),(20,200,2,2,0)],
                        validation=[(10,300,4,3,1),(20,100,1,3,0)],
                        test=[(10,300,5,4,1),(10,200,1,4,0),(20,100,4,4,1),(20,200,2,4,0)])
            for split, values in rows.items():
                with (root/f'{split}.csv').open('w', newline='') as f:
                    w=csv.writer(f);w.writerow(['user_id','item_id','score','timestamp','label']);w.writerows(values)
            # Evaluation fixture for BPR needs both classes in validation for one user.
            with (root/'validation.csv').open('a') as f:
                f.write('10,100,1,3,0\n')
            feature_path=root/'features.pt'
            torch.save(dict(item_ids=[300,100,200],features=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])),feature_path)
            for model in ('mf','ncf'):
                for loss in ('bce','mse','bpr'):
                    for mode in ('real','random'):
                        args=parser().parse_args(['--model',model,'--loss',loss,'--neg-mode',mode,
                            '--data-dir',str(root),'--item-features',str(feature_path),
                            '--output-dir',str(root/f'{model}-{loss}-{mode}'),'--epochs','1'])
                        run(args)
                        saved=torch.load(args.output_dir/'best.pt',weights_only=True)
                        self.assertIn('item_features.weight',saved['model_state'])
            # A separate non-overlapping ranking fixture demonstrates independent reload.
            with (root/'validation.csv').open('w') as f:
                f.write('user_id,item_id,score,timestamp,label\n10,200,1,3,0\n20,200,1,3,0\n')
            feature_path.unlink()
            result=evaluate(root/'mf-bce-real/best.pt',root)
            self.assertEqual(result['evaluated_users'],2)
            self.assertEqual(result['Recall@5'],1.)


if __name__ == '__main__':
    unittest.main()
