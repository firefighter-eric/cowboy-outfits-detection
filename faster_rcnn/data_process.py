from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm

labels = {87, 131, 318, 588, 1034}
labels = sorted(list(labels))
label_idx = {l: i for i, l in enumerate(sorted(list(labels)))}


class CocoDataLoader:
    def __init__(self):
        root_path = '../data/images'
        ann_path = '../data/train.json'
        coco_det = datasets.CocoDetection(root=root_path, annFile=ann_path, transform=ToTensor())

        # test_df = pd.read_csv('../data/valid.csv')
        # dev_ids = set(test_df.id.to_list())

        self.L = len(coco_det)
        train_size = int(self.L * 0.9)
        dev_size = self.L - train_size
        train_set, dev_set = torch.utils.data.random_split(coco_det, [train_size, dev_size],
                                                           torch.Generator().manual_seed(42))
        self.train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0,
                                                             collate_fn=self.collate_fn_coco)

        self.dev_data_loader = torch.utils.data.DataLoader(dev_set, batch_size=8, shuffle=False, num_workers=0,
                                                           collate_fn=self.collate_fn_coco)

    @staticmethod
    def _get_labels(dataset):
        _labels = set()
        for img, targets in tqdm(dataset):
            for target in targets:
                _labels.add(target['category_id'])
        return _labels

    @staticmethod
    def collate_fn_coco(batch) -> Tuple:
        """

        Args:
            batch: n lines of data (image, target)
            out: a batch of n data (images, targets)
        Returns:

        """

        def process_targets(_targets):
            out = []
            for target in _targets:
                # print(target)
                boxes = []
                for t in target:
                    x1, y1, w, h, = t['bbox']
                    boxes.append([x1, y1, x1 + w, y1 + h])
                boxes = torch.tensor(boxes)
                labels = torch.tensor([label_idx[t['category_id']] for t in target], dtype=torch.int64)
                out.append({'boxes': boxes,
                            'labels': labels})
            return out

        images, targets = zip(*batch)
        return images, process_targets(targets)
