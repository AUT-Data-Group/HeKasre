import json
import operator
from pathlib import Path
import fasttext
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import DataLoader


class HeKasreDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path=None, test_path=None, fasttext_path=None, mode="list", batch_size=32, num_workers=1):
        super().__init__(self)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode

        if fasttext_path is not None:
            self.fasttext_path = fasttext_path
        if train_path is not None:
            self.train_data = list(self._load_data(train_path))
        if test_path is not None:
            self.test_data = list(self._load_data(test_path))
        if val_path is not None:
            self.val_data = list(self._load_data(val_path))

    def _load_data(self, path):
        with path.open() as f:
            for item in json.load(f):
                yield self.extract_features(item["words"], item["tags"])

    def _get_embedding(self):
        if not hasattr(self, "_embedding"):
            self._embedding = fasttext.load_model(str(self.fasttext_path))
        return self._embedding

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_embedding"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def create_dataloader(self, data, shuffle=False):
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.val_data)

    def test_dataloader(self):
        return self.create_dataloader(self.test_data)

    def extract_features(self, sentence, labels=None):
        if labels is not None:
            assert len(sentence) == len(labels)

        embedding = self._get_embedding()

        features = {
            "length": len(sentence),
            "sentence": sentence,
            "embeddings": torch.FloatTensor(
                np.stack([embedding.get_word_vector(w) for w in sentence])
            ),
            "ends_with_he": torch.FloatTensor(
                [[w.endswith("ه")] for w in sentence]
            ),
        }

        if labels is not None:
            mapping = {0: 0, -1: 1, 1: 2}
            features["labels"] = torch.LongTensor([mapping[l] for l in labels])

        return features

    def collate_fn(self, batch):
        batch_data = sorted(batch, key=operator.itemgetter("length"), reverse=True)

        batch = dict(
            lengths=[d["length"] for d in batch_data],
            sentences=[d["sentence"] for d in batch_data],
            embeddings=[d["embeddings"] for d in batch_data],
            ends_with_he=[d["ends_with_he"] for d in batch_data],
            labels=[d["labels"] for d in batch_data],
        )

        if self.mode == "pack":
            batch["embeddings"] = pack_sequence(batch["embeddings"])
            batch["ends_with_he"] = pack_sequence(batch["ends_with_he"])
            batch["labels"] = pack_sequence(batch["labels"])
        else:
            assert self.mode == "list"

        return batch
