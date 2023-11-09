# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os
from typing import List, Tuple, Union, Dict

import numpy as np
import pandas

from paddle3d import transforms as T
from paddle3d.datasets import BaseDataset
from paddle3d.datasets.apollo.apollo_metric import ApolloMetric
from paddle3d.datasets.apollo import apollo_utils
from paddle3d.transforms import TransformABC


class ApolloDetDataset(BaseDataset):
    """
    """

    def __init__(self,
                 dataset_root: str,
                 dataset_list: Union[str, List[str]] = None,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 class_names: Union[list, tuple] = None,
                 class_balanced_sampling: bool = False,
                 use_road_plane: bool = False,
                 distance_threshold: float = 80.0,
                 create_gt_database: bool = False,
                 eval_class_map: Dict[str, str] = None):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_list = dataset_list
        self.mode = mode.lower()
        self.distance_threshold = distance_threshold
        self.create_gt_database = create_gt_database
        self.eval_class_map = eval_class_map
        self.class_names = class_names
        self.use_road_plane = use_road_plane
        self.dirname = 'testing' if self.is_test_mode else 'training'

        for transform in transforms:
            if isinstance(transform, T.samplingV2.SamplingDatabaseV2):
                assert transform.class_names == self.class_names, \
                    "dataset's class_name must be same as SamplingDatabaseV2"
        
        if isinstance(transforms, list):
            transforms = T.Compose(transforms)
        self.transforms = transforms

        if self.mode not in ['train', 'val', 'trainval', 'test']:
            raise ValueError(
                "mode should be 'train', 'val', 'trainval' or 'test', but got {}."
                .format(self.mode))

        self.imagesets = []
        for dataset_name in self.dataset_list:
            self.imagesets.append(os.path.join(self.dataset_root, dataset_name,
                                  'ImageSets', '{}.txt'.format(self.mode)))
        
        self.data = []
        for dataset_name, split_path in zip(self.dataset_list, self.imagesets):
            with open(split_path) as file:
                data_list = file.read().strip('\n').split('\n')
                for line in data_list:
                    line = dataset_name + '/' + line
                    self.data.append(line)
            assert self.data != [], 'the data list is empty!'

        if class_balanced_sampling and self.mode.lower() == 'train' and len(
                self.class_names) > 1:
            cls_dist = {class_name: [] for class_name in self.class_names}
            for index in range(len(self.data)):
                file_idx = self.data[index]
                kitti_records, ignored_kitti_records = self.load_annotation(
                    index)
                gt_names = []
                for anno in kitti_records:
                    class_name = anno[0]
                    if class_name in self.class_names:
                        gt_names.append(class_name)
                for class_name in set(gt_names):
                    cls_dist[class_name].append(file_idx)

            num_balanced_samples = sum([len(v) for k, v in cls_dist.items()])
            num_balanced_samples = max(num_balanced_samples, 1)
            balanced_frac = 1.0 / len(self.class_names)
            fracs = [len(v) / num_balanced_samples for k, v in cls_dist.items()]
            sampling_ratios = [balanced_frac / frac for frac in fracs]

            resampling_data = []
            for samples, sampling_ratio in zip(
                    list(cls_dist.values()), sampling_ratios):
                resampling_data.extend(samples)
                if sampling_ratio > 1.:
                    resampling_data.extend(
                        np.random.choice(
                            samples,
                            int(len(samples) * (sampling_ratio - 1.))).tolist())
            self.data = resampling_data
        self.use_road_plane = use_road_plane

    def __len__(self):
        return len(self.data)

    def load_annotation(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        filename = '{}.txt'.format(self.data[index])
        filename = os.path.join(self.dataset_root, filename.split('/')[0],
                                self.dirname, 'label_2', filename.split('/')[1])
        with open(os.path.join(filename), 'r') as csv_file:
            df = pandas.read_csv(csv_file, sep=' ', header=None)
            array = np.array(df)
            rows = []
            ignored_rows = []
            for row in array:
                # if create gt database, do not filter by class
                if self.create_gt_database:
                    rows.append(row)
                else:
                    row[0] = apollo_utils.map_class(row[0])
                    if row[0] in self.class_names:
                        rows.append(row)
                    else:
                        ignored_rows.append(row)

        kitti_records = np.array(rows)
        ignored_kitti_records = np.array(ignored_rows)
        return kitti_records, ignored_kitti_records

    @property
    def metric(self):
        gt = []
        for idx in range(len(self)):
            annos = self.load_annotation(idx)
            anno = self.FilterGTOutsideRange(annos[0])
            ignored_anno = self.FilterGTOutsideRange(annos[1])
            if len(anno) > 0 and len(ignored_anno) > 0:
                gt.append(np.concatenate((anno, ignored_anno), axis=0))
            elif len(anno) > 0:
                gt.append(anno)
            else:
                gt.append(ignored_anno)
        return ApolloMetric(
            groundtruths=gt,
            classmap={i: name
                      for i, name in enumerate(self.class_names)},
            indexes=self.data,
            eval_class_map=self.eval_class_map)
    
    def FilterGTOutsideRange(self, annos):
        if len(annos) > 0:
            mask = (annos[:, -4] >= -self.distance_threshold) & \
                   (annos[:, -4] <= self.distance_threshold) & \
                   (annos[:, -3] >= -self.distance_threshold) & \
                   (annos[:, -3] <= self.distance_threshold)
            annos = annos[mask]
        return annos

    @property
    def name(self) -> str:
        return "Apollo"

    @property
    def labels(self) -> List[str]:
        return self.class_names