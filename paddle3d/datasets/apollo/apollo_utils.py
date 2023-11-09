import numpy as np
from typing import List, Tuple

from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode

import warnings

def assess_apollo_object_difficulties(kitti_records: np.ndarray,
                                    distances_thres: List = [20, 50]):
    # 0~20m: easy, 20~50m: moderate, 50m~: hard
    num_objects = kitti_records.shape[0]
    if num_objects == 0:
        return np.full((num_objects, ), -1, dtype=np.int32)
    distances = np.sqrt((np.square(kitti_records[:, 11]) +
                         np.square(kitti_records[:, 12])).astype(float))

    easy_mask = np.ones((num_objects, ), dtype=bool)
    moderate_mask = np.ones((num_objects, ), dtype=bool)
    hard_mask = np.ones((num_objects, ), dtype=bool)

    easy_mask[np.where(distances >= distances_thres[0])] = False
    moderate_mask[np.where(distances >= distances_thres[1])] = False
    hard_mask[np.where(distances <= distances_thres[0])] = False

    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    difficulties = np.full((num_objects, ), -1, dtype=np.int32)
    difficulties[is_hard] = 2
    difficulties[is_moderate] = 1
    difficulties[is_easy] = 0

    return difficulties

# lidar record fields
# type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, dw, dl, dh, lx, ly, lz, rz
def lidar_record_to_object(
        kitti_records: np.ndarray, show_warn: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    if kitti_records.shape[0] == 0:
        if show_warn:
            warnings.warn("fake object!")
        bboxes_2d = BBoxes2D(np.zeros([0, 4]))
        bboxes_3d = BBoxes3D(
            np.zeros([0, 7]),
            origin=[0.5, 0.5, 0.],
            coordmode=CoordMode.KittiLidar,
            rot_axis=2)
        cls_names = []
    else:
        centers = kitti_records[:, 11:14]
        dims = kitti_records[:, 8:11]
        yaws = kitti_records[:, 14:15]
        bboxes_3d = BBoxes3D(
            np.concatenate([centers, dims, yaws], axis=1),
            origin=[0.5, 0.5, 0.],
            coordmode=CoordMode.KittiLidar,
            rot_axis=2)
        bboxes_2d = BBoxes2D(kitti_records[:, 4:8])
        cls_names = kitti_records[:, 0]

    return bboxes_2d, bboxes_3d, cls_names

def map_class(src_class: str) -> str:
    if src_class.lower() not in class_information:
        warnings.warn(
            "Unknown class: {} ".format(src_class)
        )
        return src_class
    else:
        return class_information[src_class.lower()]['map_class']

class_information = {
    # smallMot
    'smallmot': {'map_class': 'smallMot', 'difficulty_threshold': [20, 40]},
    'midmot': {'map_class': 'smallMot', 'difficulty_threshold': [20, 40]},
    'smallcar': {'map_class': 'smallMot', 'difficulty_threshold': [20, 40]},
    'smallvehicle': {'map_class': 'smallMot', 'difficulty_threshold': [20, 40]},

    # bigMot
    'bigmot': {'map_class': 'bigMot', 'difficulty_threshold': [30, 60]},
    'verybigmot': {'map_class': 'bigMot', 'difficulty_threshold': [30, 60]},
    'truck': {'map_class': 'bigMot', 'difficulty_threshold': [30, 60]},
    'van': {'map_class': 'bigMot', 'difficulty_threshold': [30, 60]},
    'bus': {'map_class': 'bigMot', 'difficulty_threshold': [30, 60]},
    'bigvehicle': {'map_class': 'bigMot', 'difficulty_threshold': [30, 60]},

    # pedestrian
    'pedestrian': {'map_class': 'pedestrian', 'difficulty_threshold': [10, 20]},
    'cluster': {'map_class': 'pedestrian', 'difficulty_threshold': [10, 20]},

    # nonMot
    'nonMot': {'map_class': 'nonMot', 'difficulty_threshold': [15, 30]},
    'bicyclist': {'map_class': 'nonMot', 'difficulty_threshold': [15, 30]},
    'motorcyclist': {'map_class': 'nonMot', 'difficulty_threshold': [15, 30]},
    'onlybicycle': {'map_class': 'nonMot', 'difficulty_threshold': [10, 20]},
    'motorcycle':  {'map_class': 'nonMot', 'difficulty_threshold': [15, 30]},
    'bicycle': {'map_class': 'nonMot', 'difficulty_threshold': [10, 20]},
    'cyclist': {'map_class': 'nonMot', 'difficulty_threshold': [10, 20]},
    'onlytricycle': {'map_class': 'nonMot', 'difficulty_threshold': [20, 35]},
    'tricyclist': {'map_class': 'nonMot', 'difficulty_threshold': [20, 35]},
    
    # TrafficCone
    'trafficcone': {'map_class': 'TrafficCone', 'difficulty_threshold': [8, 15]},
    'safetybarrier': {'map_class': 'TrafficCone', 'difficulty_threshold': [15, 25]},
    'sign': {'map_class': 'TrafficCone', 'difficulty_threshold': [8, 15]},
    'crashbarrel': {'map_class': 'TrafficCone', 'difficulty_threshold': [10, 20]},

    # others
    'stopbar': {'map_class': 'stopBar', 'difficulty_threshold': [8, 15]},
    'spike': {'map_class': 'spike', 'difficulty_threshold': [4, 8]},
    'smallmovable': {'map_class': 'smallMovable', 'difficulty_threshold': [8, 15]},
    'smallunmovable': {'map_class': 'smallUnmovable', 'difficulty_threshold': [8, 15]},
    'unknown': {'map_class': 'unknown', 'difficulty_threshold': [8, 15]},
    'unknow': {'map_class': 'unknow', 'difficulty_threshold': [8, 15]},
    'others': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'other': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'accessory': {'map_class': 'accessory', 'difficulty_threshold': [10, 20]},
    'wheelbarrow': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'blend': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'peopleslightly': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'vehicleslightly': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'otherslightly': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'unknownunmovable': {'map_class': 'others', 'difficulty_threshold': [8, 15]},
    'unknownmovable': {'map_class': 'others', 'difficulty_threshold': [8, 15]}
}