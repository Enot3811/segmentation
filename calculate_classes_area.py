from pathlib import Path

from tqdm import tqdm
import numpy as np

from segmentation_dataset import SegmentationDataset
from dataset_parameters.original_parameters import cls_names


def main():
    dset = SegmentationDataset(img_dir, mask_dir)
    cls_areas = {name: 0 for name in cls_names}

    for i in tqdm(range(len(dset)), 'Calculate classes areas'):
        img, mask = dset[i]
        for j, name in enumerate(cls_names):
            if name == 'unlabeled':
                continue
            cls_areas[name] += (mask == j).sum()
    total = sum(list(cls_areas.values()))
    print(cls_areas)
    print(total)

    # Calculate weights
    cls_weights = {}
    mu = 1.0
    for name in cls_areas:
        cls_weights[name] = np.log(mu * total / cls_areas[name])
    print(cls_weights)
    

if __name__ == '__main__':
    img_dir = Path(
        '/home/pc0/projects/segmentation/data/ign/images/validation')
    mask_dir = Path(
        '/home/pc0/projects/segmentation/data/ign/annotations/validation')
    main()
