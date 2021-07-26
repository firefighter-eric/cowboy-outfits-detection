# %%

from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from modeling import get_fasterrcnn_model_for_cowboy

# %%
# data

root_path = '../data/images'
ann_path = '../data/train.json'
coco_det = datasets.CocoDetection(root=root_path, annFile=ann_path, transform=ToTensor())

L = len(coco_det)
train_size = int(L * 0.9)
dev_size = L - train_size
train_set, dev_set = torch.utils.data.random_split(coco_det, [train_size, dev_size])

# %%
# labels

# from tqdm import tqdm
#
# labels = set()
# for img, targets in tqdm(iter(coco_det)):
#     for target in targets:
#         labels.add(target['category_id'])
# labels

labels = {87, 131, 318, 588, 1034}
labels = sorted(list(labels))
label_idx = {l: i for i, l in enumerate(sorted(list(labels)))}


# %%
def process_targets(targets):
    out = []
    for target in targets:
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


def collate_fn_coco(batch) -> Tuple:
    """

    Args:
        batch: n lines of data (image, target)
        out: a batch of n data (images, targets)
    Returns:

    """
    images, targets = zip(*batch)
    out = images, process_targets(targets)
    return out


train_data_loader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn_coco)

images, targets = next(iter(train_data_loader))

dev_data_loader = torch.utils.data.DataLoader(
    dev_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_coco)


# %%

# - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
#   ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
# - labels (Int64Tensor[N]): the class label for each ground-truth box


# %%

def test_model():
    model = get_fasterrcnn_model_for_cowboy()

    # For Training
    model.train()
    output = model(images, targets)  # Returns losses and detections

    # For inference
    model.eval()
    predictions = model(images)
    return output, predictions


# o, p = test_model()


# %% md
device = 'cuda:0'
# device = 'cpu'

model = get_fasterrcnn_model_for_cowboy()
model.to(device)
model.train()
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005,
#                             momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)

optimizer = torch.optim.Adam(params)


# and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)
#
# # let's train it for 10 epochs

def train_one_epoch(model, optimizer, train_data, dev_data):
    for i, (images, targets) in enumerate(tqdm(iter(train_data))):
        images = [_.to(device) for _ in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        model.zero_grad()
        outputs = model(images, targets)
        loss = sum(outputs.values())
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        if i % 100 == 0:
            print('\ntrain loss:', float(loss))
            print('dev loss:', eval(model, dev_data))


def eval(model, dev_data) -> float:
    loss = 0
    data_size = len(dev_data)
    for images, targets in iter(dev_data):
        images = [_.to(device) for _ in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        model.zero_grad()
        outputs = model(images, targets)
        loss += float(sum(outputs.values()))
    return loss / data_size


num_epochs = 3

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_data_loader, dev_data_loader)
    torch.save(model, f'../outputs/m{epoch}.pt')
