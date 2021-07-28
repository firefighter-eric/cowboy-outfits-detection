import torch
from tqdm import tqdm

from data_process import CocoDataLoader

cdl = CocoDataLoader()
dev_dataloader = cdl.dev_data_loader

model_path = '../outputs/m2.pt'
model = torch.load(model_path)

device = 'cuda:0'

model.to(device)
model.eval()

predictions = []
for imgs, targets in tqdm(dev_dataloader):
    imgs = [_.to(device) for _ in imgs]
    with torch.no_grad():
        preds = model(imgs)
    preds = [{k: v.to('cpu') for k, v in pred.items()} for pred in preds]
    predictions.append(preds)
