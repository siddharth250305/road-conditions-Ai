"""PyTorch dataset over the saved RSCD split lists, with torchvision v2 transforms."""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from roadcond.data.rscd import Record

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, size: Sequence[int], strength: float = 1.0) -> v2.Compose:
    """Resize to size = (height, width) and normalise; train also flips and jitters colour.

    Colour is a real cue for snow, ice and wet roads, so the jitter is mild and has no hue
    shift. strength scales it (0 switches it off), which the later sweep (C7) can vary.
    """
    ops: list[v2.Transform] = [v2.ToImage(), v2.Resize(tuple(size), antialias=True)]
    if train:
        ops.append(v2.RandomHorizontalFlip())
        if strength > 0:
            ops.append(
                v2.ColorJitter(
                    brightness=0.2 * strength, contrast=0.2 * strength, saturation=0.1 * strength
                )
            )
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops)


class RoadDataset(Dataset):
    """(image tensor, class index) pairs for a list of split records."""

    def __init__(
        self,
        records: Iterable[Record],
        root: str | Path,
        classes: Sequence[str],
        transform: v2.Compose | None = None,
    ):
        self.records = list(records)
        self.root = Path(root)
        self.classes = list(classes)
        self.transform = transform
        unknown = {r.friction for r in self.records} - set(self.classes)
        if unknown:
            raise ValueError(f"Records have classes {sorted(unknown)} not in {self.classes}")
        self.targets = [self.classes.index(r.friction) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.root / self.records[index].path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def load_raw(self, index: int, size: Sequence[int] | None = None) -> np.ndarray:
        """RGB uint8 array (H, W, 3), resized to size = (height, width) if given."""
        with Image.open(self.root / self.records[index].path) as img:
            image = img.convert("RGB")
        if size is not None:
            image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
        return np.asarray(image)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """Loader with a seeded shuffle; a shuffled (train) loader drops the last partial batch."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # BatchNorm cannot train on a batch of one
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
