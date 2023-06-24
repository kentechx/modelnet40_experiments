import os
import fire
from pprint import pprint
import taichi as ti
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from einops import rearrange
from typing import Literal

from pointnet import PointNet2SSGCls, PointNet2MSGCls
from dataset.modelnet import ModelNet40

ti.init(ti.cuda)


class LitModel(pl.LightningModule):
    def __init__(self, n_points, model, dropout, lr, batch_size, epochs, warm_up, optimizer):
        super().__init__()
        self.save_hyperparameters()
        self.warm_up = warm_up
        self.lr = lr
        self.batch_size = batch_size

        if model == "ssg":
            self.net = PointNet2SSGCls(3, 40, dropout=dropout)
        elif model == "msg":
            self.net = PointNet2MSGCls(3, 40, dropout=dropout)
        else:
            raise NotImplementedError

        # metrics: OA
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=40)
        self.val_acc = torchmetrics.Accuracy('multiclass', num_classes=40)

    def forward(self, x, xyz):
        return self.net(x, xyz)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, 'b n d -> b d n')
        pred = self(x, x[:, :3, :].clone())
        loss = F.cross_entropy(pred, y)
        self.train_acc(pred, y)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, 'b n d -> b d n')
        pred = self(x, x[:, :3, :].clone())
        loss = F.cross_entropy(pred, y)
        self.val_acc(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=self.trainer.estimated_stepping_batches, max_lr=self.lr,
            pct_start=self.warm_up / self.trainer.max_epochs, div_factor=10, final_div_factor=100)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        H = self.hparams
        return DataLoader(ModelNet40(n_points=H.n_points, train=True), batch_size=H.batch_size, num_workers=4,
                          shuffle=True, pin_memory=True)

    def val_dataloader(self):
        H = self.hparams
        return DataLoader(ModelNet40(n_points=H.n_points, train=False), batch_size=H.batch_size, num_workers=4,
                          shuffle=False, pin_memory=True)


def run(n_points=1024,
        model: Literal['ssg', 'msg'] = 'ssg',
        lr=1e-3,
        epochs=100,
        batch_size=32,
        warm_up=10,
        optimizer='adamw',
        dropout=0.5,
        gradient_clip_val=0,
        version='pointnet2_ssg',
        offline=False):
    # print all hyperparameters
    pprint(locals())
    pl.seed_everything(42)

    os.makedirs('wandb', exist_ok=True)
    logger = WandbLogger(project='modelnet40_experiments', name=version, save_dir='wandb', offline=offline)
    model = LitModel(n_points=n_points, model=model, dropout=dropout, batch_size=batch_size, epochs=epochs, lr=lr,
                     warm_up=warm_up, optimizer=optimizer)
    callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(logger=logger, accelerator='cuda', max_epochs=epochs, callbacks=[callback],
                         gradient_clip_val=gradient_clip_val)
    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(run)
