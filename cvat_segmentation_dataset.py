from typing import List, Dict, Union, Any, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import cv2


class CvatSegmentationDataset:

    def __init__(self, dset_pth: Union[Path, str]) -> None:
        if isinstance(dset_pth, str):
            self.dset_pth = Path(dset_pth)
        else:
            self.dset_pth = dset_pth
        annots = ET.parse(self.dset_pth / 'annotations.xml').getroot()
        
        # Get labels
        self._label_to_color = {}
        self._labels = []
        self._label_to_idx = {}
        labels_annots = annots.findall('meta/job/labels/label')
        for i, label_annot in enumerate(labels_annots):
            name = label_annot.find('name').text
            hex_color = label_annot.find('color').text
            color = [int(hex_color[j:j + 2], 16)
                     for j in range(1, len(hex_color), 2)]
            self._label_to_color[name] = color
            self._labels.append(name)
            self._label_to_idx[name] = i

        # Get images
        self.images: List[Dict[str, Any]] = []
        imgs_annots = annots.findall('image')
        for img_annots in imgs_annots:
            name = img_annots.get('name')
            shape = (int(img_annots.get('height')),
                     int(img_annots.get('width')))
            polygons = img_annots.findall('polygon')
            labels: List[int] = []
            polygons_pts: List[List[Tuple[int, int]]] = []
            for polygon in polygons:
                label = polygon.get('label')
                label_idx = self._label_to_idx[label]
                labels.append(label_idx)
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.images[idx]
    
    def get_labels(self) -> List[str]:
        """Get a list of dataset's labels

        Returns
        -------
        List[str]
            The list of the labels.
        """
        return self._labels

    def get_labels_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get labels with corresponding colors.

        Returns
        -------
        Dict[str, Tuple[int, int, int]]
            Dict that contains labels as keys and "color" as values.
        """
        return self._label_to_color
    
    def create_segmentation_masks(self, save_path: Path = None):
        """Create label masks of dataset's annotations.

        Parameters
        ----------
        save_path : Path, optional
            Path to save dir.
            If not provided then masks will be save in dataset's directory.
        """
        if save_path is None:
            save_path = self.dset_pth / 'masks'
        for image_annots in self.images:
            h, w = image_annots['shape']
            polygons = image_annots['polygons']
            labels = image_annots['labels']
            name = image_annots['name']
            canvas = np.zeros((h, w), np.uint8)
            for polygon, label in zip(polygons, labels):
                cv2.fillPoly(canvas, [np.array(polygon)], [label])
            npy_name = name.split('.')[0] + '.npy'
            np.save(save_path / npy_name, canvas)
        
    def create_color_masks(self, save_path: Path = None):
        """Create rgb masks of dataset's annotations.

        Parameters
        ----------
        save_path : Path, optional
            Path to save dir.
            If not provided then masks will be save in dataset's directory.
        """
        if save_path is None:
            save_path = self.dset_pth / 'color_masks'
        for image_annots in self.images:
            h, w = image_annots['shape']
            polygons = image_annots['polygons']
            labels = image_annots['labels']
            name = image_annots['name']
            canvas = np.zeros((h, w, 3), np.uint8)
            for polygon, label in zip(polygons, labels):
                cv2.fillPoly(canvas, [np.array(polygon)],
                             self.get_labels_colors()[label])
            np.save(save_path / name)


dset = CvatSegmentationDataset('/home/pc0/projects/segmentation/data/test/')
dset.create_segmentation_masks()
dset.create_color_masks()
