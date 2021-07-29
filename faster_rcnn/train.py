import torch
from tqdm import tqdm

from data_process import CocoDataLoader
from modeling import get_fasterrcnn_model_for_cowboy


# - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
#   ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
# - idx2label (Int64Tensor[N]): the class idx2label for each ground-truth box


def test_model(images, targets):
    model = get_fasterrcnn_model_for_cowboy()

    # For Training
    model.train()
    output = model(images, targets)  # Returns losses and detections

    # For inference
    model.eval()
    predictions = model(images)
    return output, predictions


class Trainer:
    def __init__(self, model, num_epochs, device, train_data, dev_data):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.dev_data = dev_data

    def train(self):
        self.model.to(DEVICE)
        self.model.train()
        # construct an optimizer
        params = [p for p in faster_rcnn.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # optimizer = torch.optim.Adam(params)

        for epoch in range(self.num_epochs):
            self.train_one_epoch(self.model, optimizer, lr_scheduler, self.train_data, self.device)
            torch.save(self.model, f'../outputs/models/m2/m{epoch}.pt')
            eval_loss = self.eval(faster_rcnn, self.dev_data, self.device)
            print(eval_loss)

    @staticmethod
    def train_one_epoch(model, optimizer, lr_scheduler, train_data, device):
        for i, (images, targets) in enumerate(tqdm(train_data)):
            images = [_.to(device) for _ in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i % 100 == 0:
                print('\ntrain loss:', float(loss))
                print('\ntrain loss dict:', loss_dict)

    @staticmethod
    def eval(model, dev_data, device) -> float:
        loss = 0
        data_size = len(dev_data)
        with torch.no_grad():
            for images, targets in tqdm(dev_data):
                images = [_.to(device) for _ in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                outputs = model(images, targets)
                loss += float(sum(outputs.values()))
        return loss / data_size


if __name__ == '__main__':
    DEVICE = 'cuda:0'

    cdl = CocoDataLoader()
    train_data_loader = cdl.train_data_loader
    dev_data_loader = cdl.dev_data_loader

    faster_rcnn = get_fasterrcnn_model_for_cowboy()
    trainer = Trainer(model=faster_rcnn,
                      num_epochs=10,
                      device=DEVICE,
                      train_data=train_data_loader,
                      dev_data=dev_data_loader)
    trainer.train()
