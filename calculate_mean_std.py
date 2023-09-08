"""Script to calculate dataset channel wise std and mean values."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from image_utils import read_image


def main():
    img_pths = list(DSET_DIR.glob('*.png'))

    total_mean = np.zeros((3,), dtype=np.float32)
    total_std = np.zeros((3,), dtype=np.float32)
    desc = 'Calculate mean and std'
    for img_pth in tqdm(img_pths, desc=desc):
        img = read_image(img_pth)
        img = img.astype(np.float32) / 255
        total_mean += np.mean(img, (0, 1))
        total_std += np.std(img, (0, 1))
    total_mean /= len(img_pths)
    total_std /= len(img_pths)
    print(f'mean = {total_mean}, std = {total_std}')


if __name__ == '__main__':
    # mean = [0.33437088 0.3806181  0.30854824]
    # std = [0.13211827 0.11649232 0.10825609]
    DSET_DIR = Path('/home/pc0/projects/segmentation/data/ign/images/training')
    main()
