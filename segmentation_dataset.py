from pathlib import Path
from typing import Callable, Tuple

from torch.utils.data import Dataset

from image_utils import read_image


class SegmentationDataset(Dataset):
    def __init__(
        self, img_dir: Path, anns_dir: Path, transforms: Callable = None
    ) -> None:
        super().__init__()
        self.img_pths = list(img_dir.glob('*.png'))
        self.anns_pths = list(anns_dir.glob('*.png'))
        self.img_pths.sort()
        self.anns_pths.sort()
        self.transforms = transforms

    def __getitem__(self, index: int):
        img = read_image(self.img_pths[index])
        annots = read_image(self.anns_pths[index])[..., 0]  # (h,w,3) -> (h,w)
        if self.transforms:
            augmented = self.transforms(image=img, mask=annots)
            img = augmented['image']
            annots = augmented['mask']
        return img, annots

    def __len__(self):
        return len(self.img_pths)
