import json
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm

idx2label = [0, 87, 131, 318, 588, 1034]
label2idx = {l: i for i, l in enumerate(sorted(list(idx2label)))}


class CocoDataLoader:
    """
    - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
      ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
    - idx2label (Int64Tensor[N]): the class idx2label for each ground-truth box
    """

    def __init__(self):
        root_path = '../data/images'
        ann_path = '../data/train.json'
        self.coco_det = datasets.CocoDetection(root=root_path, annFile=ann_path, transform=ToTensor())

        # test_df = pd.read_csv('../data/valid.csv')
        # dev_ids = set(test_df.id.to_list())
        self.idx2label = idx2label
        self.label2idx = label2idx
        self.idx2str = ['Nothing', 'belt', 'boot', 'cowboy_hat', 'jacket', 'sunglasses']

        self.L = len(self.coco_det)
        train_size = int(self.L * 0.95)
        dev_size = self.L - train_size
        self.train_set, self.dev_set = torch.utils.data.random_split(self.coco_det, [train_size, dev_size],
                                                                     torch.Generator().manual_seed(42))
        weights = self.get_small_obj_oversampling_weight(self.train_set, from_cache=True)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=sum(weights), replacement=True)

        self.train_all = torch.utils.data.DataLoader(self.coco_det, batch_size=4, shuffle=True, num_workers=0,
                                                     collate_fn=self.train_collate_fn)
        self.test_all = torch.utils.data.DataLoader(self.coco_det, batch_size=4, shuffle=True, num_workers=0,
                                                    collate_fn=self.test_collate_fn_coco)

        self.train_95 = torch.utils.data.DataLoader(self.train_set, batch_size=4, num_workers=0,
                                                    collate_fn=self.train_collate_fn, sampler=sampler)
        self.dev_05 = torch.utils.data.DataLoader(self.dev_set, batch_size=4, shuffle=False, num_workers=0,
                                                  collate_fn=self.train_collate_fn)
        self.test_05 = torch.utils.data.DataLoader(self.dev_set, batch_size=4, shuffle=False, num_workers=0,
                                                   collate_fn=self.test_collate_fn_coco)

    @staticmethod
    def _get_labels(dataset):
        _labels = set()
        for img, targets in tqdm(dataset):
            for target in targets:
                _labels.add(target['category_id'])
        return _labels

    @staticmethod
    def train_collate_fn(batch) -> Tuple:
        """

        Args:
            batch: n lines of data (image, target)
            out: a batch of n data (images, targets)
        Returns:

        """

        def process_targets(_targets):
            out = []
            for target in _targets:
                boxes = []
                for t in target:
                    x1, y1, w, h, = t['bbox']
                    boxes.append([x1, y1, x1 + w, y1 + h])
                boxes = torch.tensor(boxes)
                labels = torch.tensor([label2idx[t['category_id']] for t in target], dtype=torch.int64)
                out.append({'boxes': boxes,
                            'labels': labels,
                            })
            return out

        images, targets = zip(*batch)
        return images, process_targets(targets)

    @staticmethod
    def test_collate_fn_coco(batch) -> Tuple:
        return tuple(zip(*batch))

    @staticmethod
    def get_small_obj_oversampling_weight(data, area=96 ** 2, from_cache=False):
        if from_cache:
            with open('../data/weights.json') as fin:
                out = json.load(fin)
            return out

        out = [1] * len(data)
        for i, (_, target) in enumerate(tqdm(data)):
            for t in target:
                if t['area'] < area:
                    out[i] += 1

        with open('../data/weights.json', 'w') as fout:
            json.dump(out, fout)
        return out


if __name__ == '__main__':
    cdl = CocoDataLoader()
    test_data = cdl.test_05
    _imgs, _targets = next(iter(test_data))
    small_objs = cdl.get_small_obj_oversampling_weight(cdl.dev_set)
