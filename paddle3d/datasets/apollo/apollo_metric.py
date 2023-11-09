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

from typing import Dict, List
import warnings
import numpy as np

from paddle3d.datasets.kitti.kitti_utils import filter_fake_result
from paddle3d.datasets.metrics import MetricABC
from paddle3d.utils import box_utils
from paddle3d.sample import Sample
from paddle3d.thirdparty import apollo_eval
from paddle3d.utils.logger import logger


class ApolloMetric(MetricABC):
    def __init__(self, groundtruths: List[np.ndarray], classmap: Dict[int, str],
                 indexes: List, eval_class_map: Dict[str, str] = None):
        self.gt_annos = groundtruths
        self.predictions = []
        self.calibs = []
        self.classmap = classmap
        self.indexes = indexes
        self.eval_class_map = eval_class_map
        self.eval_class = []
        for mapped_class in self.eval_class_map.values():
            if mapped_class not in self.eval_class:
                self.eval_class.append(mapped_class)

    def _parse_gt_to_eval_format(self,
                                 groundtruths: List[np.ndarray]) -> List[dict]:
        res = []
        for idx, rows in enumerate(groundtruths):
            if rows.size == 0:
                warnings.warn("here is a val frame without gt!")
                res.append({
                    'name': np.zeros([0]),
                    'truncated': np.zeros([0]),
                    'occluded': np.zeros([0]),
                    'alpha': np.zeros([0]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.zeros([0]),
                    'score': np.zeros([0])
                })
            else:
                rows[:, 13] -= rows[:, 8] / 2
                names = []
                for name in rows[:, 0]:
                    if name in self.eval_class_map:
                        name = self.eval_class_map[name]
                    names.append(name)
                names = np.array(names, dtype=object)

                res.append({
                    'name': names,
                    'truncated': rows[:, 1].astype(np.float64),
                    'occluded': rows[:, 2].astype(np.int64),
                    'alpha': rows[:, 3].astype(np.float64),
                    'bbox': rows[:, 4:8].astype(np.float64),
                    'dimensions': rows[:, [10, 9, 8]].astype(np.float64),
                    'location': rows[:, 11:14].astype(np.float64),
                    'rotation_y': rows[:, 14].astype(np.float64)
                })

        return res

    def _parse_predictions_to_eval_format(
            self, predictions: List[Sample]) -> List[dict]:
        res = {}
        for pred in predictions:
            filter_fake_result(pred)
            id = pred.meta.id
            if pred.bboxes_3d is None:
                det = {
                    'truncated': np.zeros([0]),
                    'occluded': np.zeros([0]),
                    'alpha': np.zeros([0]),
                    'name': np.zeros([0]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.zeros([0]),
                    'score': np.zeros([0]),
                }
            else:
                num_boxes = pred.bboxes_3d.shape[0]
                output_names = [self.classmap[label] for label in pred.labels]
                if self.eval_class_map is None:
                    names = output_names
                else:
                    names = np.array(
                        [self.eval_class_map[output_name] for output_name in output_names])
                alpha = pred.get('alpha', np.zeros([num_boxes]))
                bboxes_3d = pred.bboxes_3d
                w = bboxes_3d[:, 3].copy()
                l = bboxes_3d[:, 4].copy()
                bboxes_3d[:, 3] = l
                bboxes_3d[:, 4] = w
                if bboxes_3d.origin != [.5, .5, 0]:
                    bboxes_3d[:, :3] += bboxes_3d[:, 3:6] * (
                        np.array([.5, .5, 0]) - np.array(bboxes_3d.origin))
                    bboxes_3d.origin = [.5, .5, 0]
                bboxes_3d[:, 6] = -bboxes_3d[:, 6] - np.pi/2
                bboxes_3d[:, 6] = box_utils.limit_period(bboxes_3d[:, 6], offset=0.5, period=np.pi * 2)
                bboxes_2d = np.zeros([num_boxes, 4])
                loc = bboxes_3d[:, :3]
                dim = bboxes_3d[:, 3:6]
                det = {
                    'truncated': np.zeros([num_boxes]),
                    'occluded': np.zeros([num_boxes]),
                    'alpha': alpha,
                    'bbox': bboxes_2d,
                    'name': names,
                    'dimensions': dim,
                    'location': loc,
                    'rotation_y': bboxes_3d[:, 6],
                    'score': pred.confidences,
                }

            res[id] = det

        return [res[idx] for idx in self.indexes]

    def update(self, predictions: List[Sample], ground_truths=None, **kwargs):
        """
        """
        self.predictions += predictions
        if 'calibs' in ground_truths:
            self.calibs.append(ground_truths['calibs'])

    def compute(self, verbose=False, **kwargs) -> dict:
        """
        """
        gt_annos = self._parse_gt_to_eval_format(self.gt_annos)
        dt_annos = self._parse_predictions_to_eval_format(self.predictions)

        if len(dt_annos) != len(gt_annos):
            raise RuntimeError(
                'The number of predictions({}) is not equal to the number of GroundTruths({})'
                .format(len(dt_annos), len(gt_annos)))

        metric_r40_dict = apollo_eval(
            gt_annos,
            dt_annos,
            current_classes=list(self.eval_class),
            metric_types=["bev", "3d"],
            recall_type='R40',
            z_axis=2,
            z_center=0.0)

        if verbose:
            for cls, cls_metrics in metric_r40_dict.items():
                logger.info("{}:".format(cls))
                for overlap_thresh, metrics in cls_metrics.items():
                    for metric_type, thresh in zip(["bbox", "bev", "3d"],
                                                   overlap_thresh):
                        if metric_type in metrics:
                            logger.info(
                                "{} AP_R40@{:.0%}: {:.2f} {:.2f} {:.2f}".format(
                                    metric_type.upper().ljust(4), thresh,
                                    *metrics[metric_type]))

        return metric_r40_dict