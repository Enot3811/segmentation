"""Discard image copies by comparing them to each other."""


from pathlib import Path
import shutil

from tqdm import tqdm

from image_utils import read_image


def main():
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_mask_dir.mkdir(parents=True, exist_ok=True)

    img_pths = list(src_img_dir.glob('*.png'))
    mask_pths = list(src_mask_dir.glob('*.png'))
    img_pths.sort()
    mask_pths.sort()
    img_area = 1000 * 1000
    threshold = 0.9

    pbar = tqdm(total=100, desc='Отбор уникальных изображений')
    i = 0
    while i < len(img_pths):
        img_pth = img_pths[i]
        mask_pth = mask_pths[i]
        img = read_image(img_pth)
        j = i + 1
        while j < len(img_pths):
            second_img_pth = img_pths[j]
            second_img = read_image(second_img_pth)
            comparing = (img == second_img).sum()
            # If pass threshold then save it
            if comparing / img_area > threshold:
                img_pths.pop(j)
                mask_pths.pop(j)
            else:
                j += 1
        dest_img_pth = dest_img_dir / img_pth.name
        dest_mask_pth = dest_mask_dir / mask_pth.name
        shutil.copyfile(img_pth, dest_img_pth)
        shutil.copyfile(mask_pth, dest_mask_pth)
        i += 1
        pbar.n = int(i / len(img_pths) * 100)
        pbar.refresh()
    pbar.n = 100
    pbar.refresh()
    pbar.close()
    print(f'Отобрано {len(img_pths)} изображений.')


if __name__ == '__main__':
    src_img_dir = Path('/home/pc0/projects/segmentation/data/ign/images/orig_validation')  # noqa
    src_mask_dir = Path('/home/pc0/projects/segmentation/data/ign/annotations/orig_validation')  # noqa
    dest_img_dir = Path('/home/pc0/projects/segmentation/data/ign/images/validation')  # noqa
    dest_mask_dir = Path('/home/pc0/projects/segmentation/data/ign/annotations/validation')  # noqa
    main()
