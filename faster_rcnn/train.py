import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import modeling
from configuration import Args
from data_process import CocoDataLoader


def test_model(images, targets):
    model = modeling.get_fasterrcnn_model_for_cowboy()

    # For Training
    model.train()
    output = model(images, targets)  # Returns losses and detections

    # For inference
    model.eval()
    predictions = model(images)
    return output, predictions


class Trainer:
    def __init__(self, args, model, optimizer, train_data, dev_data=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = None
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.train_data = train_data
        self.dev_data = dev_data
        self.model_out = args.model_out
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.n_iter = 0

        if not os.path.exists(self.model_out):
            os.makedirs(self.model_out)

    def train(self):
        self.model.to(self.device)
        self.model.train()

        self.eval()
        for epoch in range(self.num_epochs):
            self.train_one_epoch()
            torch.save(self.model, f'{self.model_out}/e{epoch}.pt')
            self.eval()

    def train_one_epoch(self):
        for i, (images, targets) in enumerate(tqdm(self.train_data)):
            images = [_.to(self.device) for _ in images]
            targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.n_iter % 10 == 0:
                self.writer.add_scalar('Loss/train', loss, self.n_iter)
                loss_dict = {k: round(float(v), 3) for k, v in loss_dict.items()}
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'{k}/train', v, self.n_iter)

                # print(f'\ntrain loss: {float(loss):.3}')
                # print(f'train loss dict: {loss_dict}')

            self.n_iter += 1

    def eval(self):
        loss = 0
        data_size = len(self.dev_data)
        with torch.no_grad():
            for images, targets in tqdm(self.dev_data):
                images = [_.to(self.device) for _ in images]
                targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
                outputs = self.model(images, targets)
                loss += float(sum(outputs.values()))
        loss /= data_size

        # print(f'iter: {self.n_iter}\tdev loss: {loss:.3f}')
        self.writer.add_scalar('Loss/dev', loss, self.n_iter)


def main():
    args = Args()

    cdl = CocoDataLoader()
    train_data = cdl.train_95
    dev_data = cdl.dev_05

    # model = modeling.get_retinanet_model_for_cowboy()
    # model = modeling.get_fasterrcnn_resnet50_model()
    # model = modeling.get_fasterrcnn_resnet153_model(num_classes=6, pretrained=True)
    model = torch.load('../models/faster_rcnn/m20/e1.pt')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    trainer = Trainer(args=args,
                      model=model,
                      optimizer=optimizer,
                      train_data=train_data,
                      dev_data=dev_data)
    trainer.train()


if __name__ == '__main__':
    main()
