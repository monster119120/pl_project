import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from archs.resnet import *

class ImageNetLightningModel(LightningModule):
    def __init__(
        self,
        data_path: str,
        pretrained: bool = False,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        workers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = kwargs['arch']
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        print('*' * 80)
        print(f'*************** Loading model {self.arch}')
        print('*' * 80)
        # self.model = resnet50(pretrained=self.pretrained)
        self.model = resnet50_dynamic(pretrained=self.pretrained)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(f"{prefix}_loss", loss_val, on_step=True, on_epoch=True)
        self.log(f"{prefix}_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dir = os.path.join(self.data_path, "train")
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose(
                [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
            ),
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers
        )
        return train_loader

    def val_dataloader(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_dir = os.path.join(self.data_path, "val")
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                val_dir,
                transforms.Compose(
                    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("ImageNetLightningModel")
        parser.add_argument("--arch", default="resnet18",)
        parser.add_argument("--workers", default=4, type=int, help="number of data loading workers (default: 4)")
        parser.add_argument("--batch-size", default=256, type=int)
        parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument("--weight-decay", default=1e-4, type=float)
        parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
        return parent_parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.accelerator == "ddp":
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = ImageNetLightningModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument("--data-path", metavar="DIR", type=str, help="path to dataset")
    parent_parser.add_argument(
        "-e", "--evaluate", dest="evaluate", action="store_true",
        help="evaluate model on validation set"
    )
    parent_parser.add_argument("--seed", type=int, default=42, help="seed for initializing training.")
    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler="simple", deterministic=True, max_epochs=50)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()