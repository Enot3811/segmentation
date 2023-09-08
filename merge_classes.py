"""Merge several classes into one new class."""


from pathlib import Path

import numpy as np
from tqdm import tqdm

from image_utils import read_image, save_image


def main():
    pths = list(source_dir.glob('*.png'))
    for pth in tqdm(pths, 'Generate new masks'):
        dest_pth = dest_dir / pth.name
        mask = read_image(pth)[..., 0]
        for i in merged_classes:
            for j in merged_classes[i]:
                mask[mask == j] = i
        save_image(np.stack((mask,) * 3, axis=2), dest_pth)


if __name__ == '__main__':
    source_dir = Path('/home/pc0/projects/segmentation/data/ign/annotations/validation')  # noqa
    dest_dir = Path('/home/pc0/projects/segmentation/data/ign/annotations/merged_validation')  # noqa
    merged_classes = {
        0: {0},
        1: {1},
        2: {2, 3, 4},
        3: {5},
        4: {6}
    }
    main()
