from typing import Dict, Sequence
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import GRU, Linear
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.optim import Adam
from torchmetrics import F1, Accuracy, Precision, Recall


class HeKasreGruModel(pl.LightningModule):
    def __init__(self, lr=0.001, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.linear_inp = Linear(in_features=301, out_features=100)
        self.gru = GRU(input_size=100, hidden_size=100, bidirectional=True)
        self.linear_out = Linear(in_features=100 + 200, out_features=3)
        self.metrics = {}

    def setup(self, stage: str):
        self.configure_metrics(stage)

    def configure_metrics(self, _) -> None:
        prec = Precision(num_classes=3)
        recall = Recall(num_classes=3)
        f1 = F1(num_classes=3)
        acc = Accuracy()
        self.metrics = {
            "precision": prec,
            "recall": recall,
            "accuracy": acc,
            "f1": f1,
        }

    def compute_metrics(
        self, predictions, labels, mode="val"
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions[labels != -100]
        labels = labels[labels != -100]
        return {
            f"{mode}_{k}": metric(predictions, labels)
            for k, metric in self.metrics.items()
        }

    def forward(
        self,
        embeddings: PackedSequence,
        ends_with_he: PackedSequence,
        labels: PackedSequence,
        sentences: Sequence[str] = None,
        lengths: torch.Tensor = None,
    ):
        z = embeddings
        z = z._replace(data=torch.cat([z.data, ends_with_he.data], axis=1))
        z = z._replace(data=self.linear_inp(z.data))
        z_skip = z
        z = z._replace(data=F.elu(z.data))
        z, _ = self.gru(z)
        z = z._replace(data=torch.cat([z.data, z_skip.data], axis=1))
        z = z._replace(data=self.linear_out(z.data))
        logits = z

        loss = F.cross_entropy(z.data, labels.data)

        return loss, logits

    def inference(
        self,
        embeddings: torch.Tensor,
        ends_with_he: torch.Tensor,
        sentence: Sequence[str] = None,
        length: torch.Tensor = None,
    ):
        # embedding: [L, 300]
        embeddings = embeddings.unsqueeze(1)
        # embedding: [L, 1, 300]

        # ends_with_he: [L, 1]
        ends_with_he = ends_with_he.unsqueeze(1)
        # ends_with_he: [L, 1, 1]

        z = torch.cat([embeddings, ends_with_he], axis=-1)
        # z: [L, 1, 301]

        z = self.linear_inp(z)
        z_skip = z
        # z: [L, 1, 100]

        z = F.elu(z)
        z, _ = self.gru(z)
        # z: [L, 1, 200]

        z = torch.cat([z, z_skip], axis=-1)
        # z: [L, 1, 300]

        z = self.linear_out(z.data)
        # z: [L, 1, 3]

        return z

    def common_step(self, prefix: str, batch) -> torch.Tensor:
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits.data, dim=1)
        metric_dict = self.compute_metrics(
            preds.data, batch["labels"].data, mode=prefix
        )
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
