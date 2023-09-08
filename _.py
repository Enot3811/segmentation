from typing import List, Dict, Union, Any, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET


class CvatSegmentationDataset:

    def __init__(self, dset_pth: Union[Path, str]) -> None:
        if isinstance(dset_pth, str):
            dset_pth = Path(dset_pth)
        annots = ET.parse(dset_pth / 'annotations.xml').getroot()
        
        # Get labels
        self.labels: List[Dict[str, Any]] = []
        labels_annots = annots.findall('meta/job/labels/label')
        for label_annot in labels_annots:
            name = label_annot.find('name').text
            hex_color = label_annot.find('color').text
            color = (int(hex_color[i:i + 2], 16)
                     for i in range(1, len(hex_color), 2))
            self.labels.append({
                'name': name,
                'color': color})

        # Get images
        self.images: List[Dict[str, Any]] = []
        imgs_annots = annots.findall('image')
        for img_annots in imgs_annots:
            name = img_annots.get('name')
            polygons = img_annots.findall('polygon')
            labels: List[str] = []
            pts: List[Tuple[int, int]] = []
            for polygon in polygons:
                labels.append(polygon.get('label'))
                pts_str = polygon.get('points')
                pts = pts_str.split(';')
                pts = list(map(
                    lambda xy: tuple(map(lambda cord: int(float(cord)),
                                         xy.split(','))),
                    pts))
            self.images.append({
                'name': name,
                'labels': labels,
                'points': pts
            })

    def __getitem__(self) -> Dict[str, Any]:
        # TODO вернуть из images дикт по индексу
        pass

    def get_labels(self) -> List[str]:
        # TODO docs
        return self.labels
    
    # TODO Написать функцию конвертации полигонов в cv2 картинку


dset = CvatSegmentationDataset('/home/pc0/projects/segmentation/data/test/')
