from tqdm import tqdm
import torch
from tqdm import tqdm

from data_process import CocoDataLoader
from plot import plot

print(torch.cuda.get_device_name(0))
# import json

cdl = CocoDataLoader()
dev_data = cdl.dev_data_loader
idx2str = cdl.idx2str

model_path = '../outputs/models/m4/m9.pt'
model = torch.load(model_path)

device = 'cuda:0'

model.to(device)
model.eval()

predictions = []
true_targets = []
images = []
for imgs, targets in tqdm(dev_data):
    images += imgs
    imgs = [_.to(device) for _ in imgs]
    with torch.no_grad():
        preds = model(imgs)
    preds = [{k: v.to('cpu') for k, v in pred.items()} for pred in preds]
    predictions += preds
    true_targets += targets

# with open('../data/predictions.json', 'w') as fout:
#     json.dump(predictions, fout)

# imgs, _ = next(iter(dev_data))
# img = imgs[0]
# img = images[0]
# pred = predictions[0]

for img, pred, true_target in zip(images, predictions, true_targets):
    plot(img, pred, true_target, idx2str, 0.5)
