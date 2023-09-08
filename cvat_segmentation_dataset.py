from typing import List, Dict, Union, Any, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import matplotlib.pyplot as plt


class CvatSegmentationDataset:

    def __init__(self, dset_pth: Union[Path, str]) -> None:
        if isinstance(dset_pth, str):
            dset_pth = Path(dset_pth)
        annots = ET.parse(dset_pth / 'annotations.xml').getroot()
        
        # Get labels
        self.label_to_color = {}
        self.labels: List[Dict[str, Any]] = []
        labels_annots = annots.findall('meta/job/labels/label')
        for label_annot in labels_annots:
            name = label_annot.find('name').text
            hex_color = label_annot.find('color').text
            color = (int(hex_color[i:i + 2], 16)
                     for i in range(1, len(hex_color), 2))
            self.label_to_color[name] = color

        # Get images
        self.images: List[Dict[str, Any]] = []
        imgs_annots = annots.findall('image')
        for img_annots in imgs_annots:
            name = img_annots.get('name')
            shape = (int(img_annots.get('height')),
                     int(img_annots.get('width')))
            polygons = img_annots.findall('polygon')
            labels: List[str] = []
            polygons_pts: List[List[Tuple[int, int]]] = []
            for polygon in polygons:
                labels.append(polygon.get('label'))
                pts_str = polygon.get('points')
                pts = pts_str.split(';')
                pts = list(map(
                    lambda xy: tuple(map(lambda cord: int(float(cord)),
                                         xy.split(','))),
                    pts))
                polygons_pts.append(pts)
            self.images.append({
                'name': name,
                'labels': labels,
                'polygons': polygons_pts,
                'shape': shape
            })
        # TODO с цветами что-то не то

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.images[idx]

    def get_labels(self) -> List[Dict[str, Any]]:
        """Get labels with corresponding colors.

        Returns
        -------
        List[Dict[str, Any]]
            List of dicts that have "name" and "color" keys.
        """
        return self.labels
    
    def create_segmentation_masks(self):
        for image_annots in self.images:
            h, w = image_annots['shape']
            polygons = image_annots['polygons']
            labels = image_annots['labels']
            name = image_annots['name']
            canvas = np.zeros((h, w, 3), np.uint8)
            for polygon, label in zip(polygons, labels):
                cv2.fillPoly(canvas, polygon, self.label_to_color[label])
                
            plt.imshow(canvas)
            plt.show()
    
        # TODO Написать функцию конвертации полигонов в cv2 картинку


dset = CvatSegmentationDataset('/home/pc0/projects/segmentation/data/test/')
dset.create_segmentation_masks()
