from collections.abc import Iterable
from typing import List, Union


def kitti_eval(gt_annos: List[dict],
               dt_annos: List[dict],
               current_classes: Union[None, int, List[int]] = (0, 1, 2),
               metric_types=("bbox", "bev", "3d"),
               recall_type='R40',
               z_axis=1,
               z_center=1.0):
    """
    """
    from paddle3d.thirdparty.kitti_object_eval_python.eval import \
        get_official_eval_result
    if not isinstance(current_classes, Iterable):
        current_classes = [current_classes]

    return get_official_eval_result(
        gt_annos,
        dt_annos,
        current_classes=current_classes,
        metric_types=metric_types,
        recall_type=recall_type,
        z_axis=z_axis,
        z_center=z_center)

def apollo_eval(gt_annos: List[dict],
               dt_annos: List[dict],
               current_classes: Union[None, int, List[int]] = (0, 1, 2),
               metric_types=("bbox", "bev", "3d"),
               recall_type='R40',
               z_axis=1,
               z_center=1.0):
    """
    """
    from paddle3d.thirdparty.apollo_object_eval_python.apollo_eval import \
        get_official_eval_result
    if not isinstance(current_classes, Iterable):
        current_classes = [current_classes]

    return get_official_eval_result(
        gt_annos,
        dt_annos,
        current_classes=current_classes,
        metric_types=metric_types,
        recall_type=recall_type,
        z_axis=z_axis,
        z_center=z_center)
