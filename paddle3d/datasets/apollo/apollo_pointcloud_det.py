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

import os
import numpy as np

from paddle3d.apis import manager
from paddle3d.datasets.apollo import apollo_utils
from paddle3d.datasets.apollo.apollo_det import ApolloDetDataset
from paddle3d.sample import Sample
from paddle3d.utils import box_utils

@manager.DATASETS.add_component
class ApolloPCDataset(ApolloDetDataset):
    """
    """

    def __getitem__(self, index: int) -> Sample:
        filename = '{}.bin'.format(self.data[index])
        path = os.path.join(self.dataset_root, filename.split('/')[0], 
                            self.dirname, 'velodyne', filename.split('/')[1])

        sample = Sample(path=path, modality="lidar")
        sample.meta.id = self.data[index]

        if self.is_train_mode:
            kitti_records, ignored_kitti_records = self.load_annotation(index)
            kitti_records = self.adjust_size_center_yaw(kitti_records)
            ignored_kitti_records = self.adjust_size_center_yaw(ignored_kitti_records)
            _, bboxes_3d, cls_names = apollo_utils.lidar_record_to_object(
                kitti_records, show_warn=True)
            _, ignored_bboxes_3d, _ = apollo_utils.lidar_record_to_object(
                ignored_kitti_records, show_warn=False)

            sample.bboxes_3d = bboxes_3d
            if self.create_gt_database:
                sample.labels = np.array(cls_names)
            else:
                sample.labels = np.array(
                    [self.class_names.index(name) for name in cls_names], dtype=np.int64)
            sample.ignored_bboxes_3d = ignored_bboxes_3d
            if self.use_road_plane:
                sample.road_plane = self.load_road_plane(index)

        if self.transforms:
            sample = self.transforms(sample)

        if 'path' not in sample:
            sample.path = path
        return sample

    def load_road_plane(self, index):
        file_name = '{}.txt'.format(self.data[index])
        plane_file = os.path.join(self.base_dir, 'planes', file_name)
        if not os.path.exists(plane_file):
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def adjust_size_center_yaw(self, kitti_records):
        if kitti_records.shape[0] == 0:
            return kitti_records
        # hwl -> wlh
        kitti_records[:, 8:11] = kitti_records[:, [9, 10, 8]]
        # geometric center -> bottom center
        kitti_records[:, 13] -= kitti_records[:, 10] / 2
        # yaw -> rotation_y
        rotation_y = -kitti_records[:, -1] - np.pi/2
        rotation_y = box_utils.limit_period(rotation_y, offset=0.5, period=np.pi * 2)
        kitti_records[:, -1] = rotation_y
        return kitti_records