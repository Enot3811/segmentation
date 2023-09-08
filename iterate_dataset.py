from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt

from segmentation_dataset import SegmentationDataset
from model_utils import mask_to_rgb, overlay_segmentation
from dataset_parameters.original_parameters import label_colors
# from dataset_parameters.merged_parameters import label_colors


def main():
    crop_size = (513, 513)

    # Get augmentations
    transforms = A.Compose([
        # A.VerticalFlip(),
        # A.HorizontalFlip(),
        # A.RandomRotate90(p=1.0),
        # A.Transpose(),
        A.Resize(*crop_size),
        # A.OneOf([
        #     A.Resize(*crop_size),
        #     A.RandomResizedCrop(*crop_size)
        # ], p=1.0),
    ])

    dset = SegmentationDataset(img_dir, mask_dir, transforms)

    for i in range(len(dset)):
        img, mask = dset[i]
        segmentation_mask = mask_to_rgb(mask, label_colors)
        overlay_img = overlay_segmentation(img, segmentation_mask)

        plt.figure(figsize=(12, 10), dpi=100)
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Image")
        plt.imshow(img)

        plt.subplot(1, 3, 2)
        plt.title("Segmentation")
        plt.axis("off")
        plt.imshow(segmentation_mask)

        plt.subplot(1, 3, 3)
        plt.title("Overlaid")
        plt.axis("off")
        plt.imshow(overlay_img)
            
        plt.show()


if __name__ == '__main__':
    img_dir = Path('/home/pc0/projects/segmentation/data/ign/images/validation')  # noqa
    mask_dir = Path('/home/pc0/projects/segmentation/data/ign/annotations/validation')  # noqa
    main()
