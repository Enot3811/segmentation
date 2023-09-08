from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
import matplotlib.pyplot as plt

from model_utils import (
    get_model, create_segmentation_map, overlay_segmentation)
from image_utils import read_image
from dataset_parameters.original_parameters import label_colors


def main():
    # Load model
    model = get_model(n_cls)
    model_weights = torch.load(model_pth)
    unloaded_keys = model.load_state_dict(model_weights, strict=False)
    model.eval()
    print('Unloaded weights:', unloaded_keys, sep='\n')

    # Get preprocessing
    resize = A.Resize(*crop_size)
    model_transf = A.Compose([
        A.Normalize(),
        ToTensor()
    ])

    # Load sample
    img = read_image(img_pth)
    img = resize(image=img)['image']
    img_to_model = model_transf(image=img)['image'][None, ...]

    # Pass through the model
    with torch.no_grad():
        out = model(img_to_model)['out']
        segmentation = create_segmentation_map(out, label_colors)[0]
    overlay_img = overlay_segmentation(img, segmentation)

    # Show results
    plt.figure(figsize=(12, 10), dpi=100)
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Image")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Segmentation")
    plt.axis("off")
    plt.imshow(segmentation)

    plt.subplot(1, 3, 3)
    plt.title("Overlaid")
    plt.axis("off")
    plt.imshow(overlay_img)
        
    plt.show()


if __name__ == '__main__':
    n_cls = 7
    model_pth = Path('work_dir/train_3/best_model.pt')
    img_pth = Path('/home/pc0/projects/segmentation/data/ign/images/validation/c79_0915_6395.png')  # noqa
    crop_size = (513, 513)
    main()
