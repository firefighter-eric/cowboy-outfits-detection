import torch
from tqdm import tqdm

from data_process import CocoDataLoader
from modeling import get_fasterrcnn_model_for_cowboy
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, model, num_epochs, device, train_data, dev_data, model_out, log_out):
        self.model = model
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.0001, weight_decay=0.00001)
        # optimizer = torch.optim.Adam(params)
        self.lr_scheduler = None
        self.device = device
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.dev_data = dev_data
        self.model_out = model_out
        self.writer = SummaryWriter(log_dir=log_out)
        self.n_iter = 0

    def train(self):
        self.model.to(DEVICE)
        self.model.train()

        self.eval()
        for epoch in range(self.num_epochs):
            self.train_one_epoch()
            torch.save(self.model, f'{self.model_out}m{epoch}.pt')
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

            if i % 100 == 0:
                self.writer.add_scalar('Loss/train', loss, self.n_iter)
                loss_dict = {k: round(float(v), 3) for k, v in loss_dict.items()}
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'{k}/train', v, self.n_iter)

                print(f'\ntrain loss: {float(loss):.3}')
                print(f'train loss dict: {loss_dict}')

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

        print(f'iter: {self.n_iter}\tdev loss: {loss:.3f}')
        self.writer.add_scalar('Loss/dev', loss, self.n_iter)


if __name__ == '__main__':
    DEVICE = 'cuda:0'

    cdl = CocoDataLoader()
    data = cdl.train_all
    # train_data = cdl.train_data_loader
    dev_data = cdl.dev_data_loader

    faster_rcnn_model = get_fasterrcnn_model_for_cowboy()
    # for name, para in faster_rcnn_model.named_parameters():
    #     if name not in {'roi_heads.box_predictor.cls_score.weight',
    #                     'roi_heads.box_predictor.cls_score.bias',
    #                     'roi_heads.box_predictor.bbox_pred.weight',
    #                     'roi_heads.box_predictor.bbox_pred.bias'}:
    #         para.requires_grad = False
    #     else:
    #         para.requires_grad = True

    for name, para in faster_rcnn_model.backbone.named_parameters():
        para.requires_grad = False

    trainer = Trainer(model=faster_rcnn_model,
                      num_epochs=10,
                      device=DEVICE,
                      train_data=data,
                      dev_data=dev_data,
                      model_out='../outputs/models/m8/',
                      log_out='../runs/m8/')
    trainer.train()
