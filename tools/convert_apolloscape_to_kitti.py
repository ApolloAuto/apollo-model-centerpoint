#!/usr/bin/env python3

"""
Convert apolloscape training data to kitti format
"""

import os
import argparse
from pypcd import pypcd
import numpy as np


names_ = ["", "00000", "0000", "000", "00", "0"]
type_mapper = {
    "1": "smallVehicle",
    "2": "bigVehicle",
    "3": "pedestrian",
    "4": "cyclist",
    "5": "trafficCone",
    "6": "other"
}

def save_pcd_file(pcd_file, output_folder, count):
    """save pcd file to dir
    """
    # bin file
    bin_name = str(count)
    if len(bin_name) < len(names_) and len(bin_name) > 0:
        bin_name = names_[len(bin_name)] + bin_name
    bin_file = os.path.join(output_folder, bin_name + '.bin')
    # read pointcloud data
    pcd = pypcd.PointCloud.from_path(pcd_file)
    data = list()
    for line in pcd.pc_data:
      d = list()
      for e in line:
        d.append(float(e))
      data.append(d)
    data = np.asarray(data, dtype=np.float32)
    data = data.reshape(-1)
    data.tofile(bin_file)  # save to binary file


def save_label_file(label_file, output_folder, count):
    """save label file to dir
    """
    # output label file
    txt_name = str(count)
    if len(txt_name) < len(names_) and len(txt_name) > 0:
        txt_name = names_[len(txt_name)] + txt_name
    txt_name = os.path.join(output_folder, txt_name + '.txt')
    f = open(label_file, 'r')
    lines = f.readlines()
    f.close()

    with open(txt_name, 'w') as f:
        for line in lines:
            bbox = line.strip().split(' ')
            obj_type = bbox[0]
            # w,l,h and theta
            width = bbox[5]
            length = bbox[4]
            height = bbox[6]
            theta = bbox[7]
            # center
            x = bbox[1]
            y = bbox[2]
            z = bbox[3]
            if obj_type in type_mapper:
                f.write(type_mapper[obj_type]); f.write(' ')
            else:
                print('type not in type_mapper')
                f.write(type_mapper['6']); f.write(' ')
            for _ in range(7):
                f.write('0'); f.write(' ')
            f.write(height); f.write(' ')
            f.write(width); f.write(' ')
            f.write(length); f.write(' ')
            f.write(x); f.write(' ')
            f.write(y); f.write(' ')
            f.write(z); f.write(' ')
            f.write(theta); f.write('\n')


def convert_to_kitti(pcd_path, label_path, output_path):
    """convert apolloscape dataset to kitti format
    """
    record_dirs = os.listdir(pcd_path)
    # create output dir
    pcd_folder = os.path.join(output_path, 'training/velodyne')
    label_folder = os.path.join(output_path, 'training/label')
    if not os.path.exists(pcd_folder):
        os.makedirs(pcd_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    # convert
    count = 0
    for dir in record_dirs:
        # pcd dir
        pcd_dir = os.path.join(pcd_path, dir)
        # label dir
        label_dir = dir[7:-6]
        label_dir = os.path.join(label_path, label_dir)
        pcds = os.listdir(pcd_dir)
        for pcd in pcds:
            # pcd file
            pcd_file = os.path.join(pcd_dir, pcd)
            label_file = os.path.join(label_dir, pcd[:-4] + '.txt')
            if os.path.isfile(pcd_file) and os.path.isfile(label_file):
                save_pcd_file(pcd_file, pcd_folder, count)
                save_label_file(label_file, label_folder, count)
                count += 1


def main(pcd_path, label_path, output_path):
    """main
    """
    convert_to_kitti(pcd_path, label_path, output_path)


if __name__ == "__main__":
    """
    pcd_path:
        |- result_9048_1_frame
        |- result_9048_3_frame
        |- ...
    label_path:
        |- 9048_1
        |- 9048_3
        |- ...
    """
    parser = argparse.ArgumentParser(description='Convert to kitti format')
    parser.add_argument('--pcd_path', type=str, default=None,
                        help='Specify the pcd path')
    parser.add_argument('--label_path', type=str, default=None,
                        help='Specify the label path')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Specify the output_path')

    args = parser.parse_args()
    main(args.pcd_path,
         args.label_path,
         args.output_path)
