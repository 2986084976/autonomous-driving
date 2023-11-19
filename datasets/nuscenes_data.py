"""
nuscenes.py
Created by zenn at 2021/9/1 15:05
"""
import os

import numpy as np
import pickle
import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_scenes

from pyquaternion import Quaternion

from datasets import points_utils, base_dataset
from datasets.data_classes import PointCloud
import time
general_to_tracking_class = {"animal": "void / ignore",
                             "human.pedestrian.personal_mobility": "void / ignore",
                             "human.pedestrian.stroller": "void / ignore",
                             "human.pedestrian.wheelchair": "void / ignore",
                             "movable_object.barrier": "void / ignore",
                             "movable_object.debris": "void / ignore",
                             "movable_object.pushable_pullable": "void / ignore",
                             "movable_object.trafficcone": "void / ignore",
                             "static_object.bicycle_rack": "void / ignore",
                             "vehicle.emergency.ambulance": "void / ignore",
                             "vehicle.emergency.police": "void / ignore",
                             "vehicle.construction": "void / ignore",
                             "vehicle.bicycle": "bicycle",
                             "vehicle.bus.bendy": "bus",
                             "vehicle.bus.rigid": "bus",
                             "vehicle.car": "car",
                             "vehicle.motorcycle": "motorcycle",
                             "human.pedestrian.adult": "pedestrian",
                             "human.pedestrian.child": "pedestrian",
                             "human.pedestrian.construction_worker": "pedestrian",
                             "human.pedestrian.police_officer": "pedestrian",
                             "vehicle.trailer": "trailer",
                             "vehicle.truck": "truck", }

tracking_to_general_class = {
    'void / ignore': ['animal', 'human.pedestrian.personal_mobility', 'human.pedestrian.stroller',
                      'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
                      'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack',
                      'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.construction'],
    'bicycle': ['vehicle.bicycle'],
    'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
    'car': ['vehicle.car'],
    'motorcycle': ['vehicle.motorcycle'],
    'pedestrian': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                   'human.pedestrian.police_officer'],
    'trailer': ['vehicle.trailer'],
    'truck': ['vehicle.truck']}


class NuScenesDataset(base_dataset.BaseDataset):
    def __init__(self, path, split, category_name="Car", version='v1.0-trainval', **kwargs):
        super().__init__(path, split, category_name, **kwargs)
        self.nusc = NuScenes(version=version, dataroot=path, verbose=False)
        self.version = version
        self.key_frame_only = kwargs.get('key_frame_only', False)
        self.min_points = kwargs.get('min_points', False)
        self.preload_offset = kwargs.get('preload_offset', -1)
        self.track_instances = self.filter_instance(split, category_name.lower(), self.min_points)#所以token来自这里track_instances
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()#tracklet_anno_list来自这里
        # print("1111111111111")
        if self.preloading:
            self.training_samples = self._load_data()

    def filter_instance(self, split, category_name=None, min_points=-1):
        """
        This function is used to filter the tracklets.

        split: the dataset split
        category_name:
        min_points: the minimum number of points in the first bbox
        """
        #所以token来自这里
        if category_name is not None:
            general_classes = tracking_to_general_class[category_name]
        instances = []
        # print("22222222222")
        scene_splits = nuscenes.utils.splits.create_splits_scenes()
        for instance in self.nusc.instance:
            anno = self.nusc.get('sample_annotation', instance['first_annotation_token'])
            sample = self.nusc.get('sample', anno['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            instance_category = self.nusc.get('category', instance['category_token'])['name']
            if scene['name'] in scene_splits[split] and anno['num_lidar_pts'] >= min_points and \
                    (category_name is None or category_name is not None and instance_category in general_classes):
                # print("scene_splits[split]",scene_splits[split])
                # print("general_classes",general_classes)#都是car
                instances.append(instance)#instance来自这里
        return instances

    def _build_tracklet_anno(self):
        #所以总归来自这里？
        # print("#所以总归来自这里？")
        list_of_tracklet_anno = []
        list_of_tracklet_len = []
        # print("333333333333")
        for instance in self.track_instances:
            track_anno = []
            curr_anno_token = instance['first_annotation_token']
            # print("instance['first_annotation_token']",instance['first_annotation_token'])
            #ann_record来自curr_anno_token
            #如果curr_anno_token不空就追
            while curr_anno_token != '':
                #所以来自sample_data_lidar和ann_record
                # print("curr_anno_token",curr_anno_token)
                ann_record = self.nusc.get('sample_annotation', curr_anno_token)
                # print("ann_record",ann_record)
                sample = self.nusc.get('sample', ann_record['sample_token'])
                #所以来自sample_data_lidar
                sample_data_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                # print("sample_data_lidar",sample_data_lidar)
                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue
                #所以来自track_anno
                track_anno.append({"sample_data_lidar": sample_data_lidar, "box_anno": ann_record})
            # print("---------------------------------")
            # print("track_anno:",track_anno)
            list_of_tracklet_anno.append(track_anno)#所以是来自这里
            list_of_tracklet_len.append(len(track_anno))
        return list_of_tracklet_anno, list_of_tracklet_len

    def _load_data(self):
        print('preloading data into memory')
        preload_data_path = os.path.join(self.path,
                                         f"preload_nuscenes_{self.category_name}_{self.split}_{self.version}_{self.preload_offset}_{self.min_points}.dat")
        # print("version",self.version)
        # if os.path.isfile(preload_data_path):
        #     # print("size:",os.path.getsize(preload_data_path))
        #     # print(f'loading from saved file {preload_data_path}.')
        #     # ta = time.time()
        #     with open(preload_data_path, 'rb') as f:
        #         print("pickle.load(f):",f)
        #         training_samples = pickle.load(f)
            # tb = time.time()
            # print("_load_data:",tb-ta)
        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
                # print("COMMON??")
                # # 写入到txt文件中，文件名与preload_data_path一致
                # txt_file_path = preload_data_path.replace('.dat', '.txt')
                # with open(txt_file_path, 'w') as txt_file:
                #     for sample in training_samples:
                #         txt_file.write(str(sample) + '\n')
        # else:
        #     print(f'File {preload_data_path} not found. Skipping...')
        # else:
        #     print('reading from annos')
        #     training_samples = []
        #     for i in range(len(self.tracklet_anno_list)):
        #         frames = []
        #         #anno来自tracklet_anno_list
        #         # print("tracklet_anno_list:",self.tracklet_anno_list)
        #         for anno in self.tracklet_anno_list[i]:
        #             #这里的anno
        #             frames.append(self._get_frame_from_anno_data(anno))
        #         training_samples.append(frames)
        #     with open(preload_data_path, 'wb') as f:
        #         print(f'saving loaded data to {preload_data_path}')
        #         pickle.dump(training_samples, f)
        else:
            print('reading from annos')
            training_samples = []
            for i in range(len(self.tracklet_anno_list)):
                frames = []
                # anno来自tracklet_anno_list
                # print("tracklet_anno_list:",self.tracklet_anno_list)
                for anno in self.tracklet_anno_list[i]:
                    try:
                        # 这里的anno
                        frames.append(self._get_frame_from_anno_data(anno))
                    except FileNotFoundError:
                        # print(f'File not found for anno: {anno}. Skipping...')
                        continue
                training_samples.append(frames)
            with open(preload_data_path, 'wb') as f:
                print(f'saving loaded data to {preload_data_path}')
                pickle.dump(training_samples, f)
                #取20%training_samples再返回，是否全
        return training_samples

    def get_num_tracklets(self):
        # print("55555555")
        return len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        # print("666666")
        return sum(self.tracklet_len_list)

    def get_num_frames_tracklet(self, tracklet_id):
        # print("777777")
        return self.tracklet_len_list[tracklet_id]

    def get_frames(self, seq_id, frame_ids):
        # print("888888888888888")
        if self.preloading:
            # print("Size of training_samples[seq_id]:", len(self.training_samples[seq_id]))
            # print("Size of frame_ids:", len(frame_ids))
            # print("Max frame_id:", max(frame_ids))
            # print("Min frame_id:", min(frame_ids))
            # print("Length of self.training_samples[seq_id]:", len(self.training_samples[seq_id]))
            #如果size小于3，就返回？
            # print("self.training_samples[seq_id]",self.training_samples[seq_id])
            # print("frame_ids",frame_ids)
            # print("---------------------------------------------------------")
            if(len(self.training_samples[seq_id]) == 0):
                # print("zerozero")
                return None
                
            frames = [self.training_samples[seq_id][f_id] for f_id in frame_ids]

        else:
            seq_annos = self.tracklet_anno_list[seq_id]
            start = time.time()
            # print("here?")
            frames = [self._get_frame_from_anno_data(seq_annos[f_id]) for f_id in frame_ids]
            end = time.time()
            # print("get_frames:",end - start)


        return frames

    def _get_frame_from_anno_data(self, anno):
        # print("99999999999")
        t1 = time.time()
        #sample_data_lidar由anno['sample_data_lidar'],不过好像是从dat文件中找到的
        # print("anno['sample_data_lidar']:",anno['sample_data_lidar'])
        sample_data_lidar = anno['sample_data_lidar']
        box_anno = anno['box_anno']
        bb = Box(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                 name=box_anno['category_name'], token=box_anno['token'])
        #pcl_path由self.path和sample_data_lidar来决定，估计sample_data_lider就是对应的n015的
        pcl_path = os.path.join(self.path, sample_data_lidar['filename'])
        #没n015，所以pcl_path应该就是n015的
        pc = LidarPointCloud.from_file(pcl_path)

        cs_record = self.nusc.get('calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        poserecord = self.nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        pc = PointCloud(points=pc.points)
        if self.preload_offset > 0:
            pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)
        t2 = time.time()
        # print("99999999999")
        # print("_get_frame_from_anno_data:",t2-t1)
        return {"pc": pc, "3d_bbox": bb, 'meta': anno}
