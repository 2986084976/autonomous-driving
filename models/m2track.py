"""
m2track.py
Created by zenn at 2021/11/24 13:10
"""
from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet

import torch
from torch import nn
import torch.nn.functional as F

from utils.metrics import estimateOverlap, estimateAccuracy
from torchmetrics import Accuracy

import numpy as np
from shapely.geometry import LineString, box
from shapely import affinity
import time
def my_iou_3d(box1, box2):
    """
    两个box均为7元素list，7个元素分别是中心点xyz坐标、箱子长宽高和偏航角（弧度制）
    """
    result_xy, result_z, result_v = [], [], []
    for b in [box1, box2]:
        # 先解包获取两框长宽高、中心坐标、偏航角
        l, w, h, x, y, z, yaw = b

        # 计算体积
        result_v.append(l * w * h)

        # 构造z轴
        ls = LineString([[0, z - h / 2], [0, z + h / 2]])
        result_z.append(ls)

        # 构造xy平面部分的矩形
        poly = box(x - l / 2, y - w / 2, x + l / 2, y + w / 2)
        poly_rot = affinity.rotate(poly, yaw, use_radians=True)
        result_xy.append(poly_rot)

    # 计算xy平面面积重叠、z轴重叠
    overlap_xy = result_xy[0].intersection(result_xy[1]).area
    overlap_z = result_z[0].intersection(result_z[1]).length

    # 计算IOU
    overlap_xyz = overlap_z * overlap_xy
    # print("np.sum(result_v) ",np.sum(result_v) )
    # print("overlap_xyz",overlap_xyz)
    return overlap_xyz / (np.sum(result_v) - overlap_xyz)


def com_box(l, w, h, x, y, z, yaw):
    half_length = l / 2
    half_width = w / 2
    half_height = h / 2
    cos_yaw = np.cos(yaw)
    # print("cos_yaw:",cos_yaw)
    sin_yaw = np.sin(yaw)

    rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0],
                                [sin_yaw, cos_yaw, 0],
                                [0, 0, 1]])
    vertices = np.array([[-half_length, -half_width, -half_height],
                         [ half_length, -half_width, -half_height],
                         [-half_length,  half_width, -half_height],
                         [ half_length,  half_width, -half_height],
                         [-half_length, -half_width,  half_height],
                         [ half_length, -half_width,  half_height],
                         [-half_length,  half_width,  half_height],
                         [ half_length,  half_width,  half_height]])
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    x_min = np.min(rotated_vertices[:, 0]) + x
    y_min = np.min(rotated_vertices[:, 1]) + y
    z_min = np.min(rotated_vertices[:, 2]) + z
    x_max = np.max(rotated_vertices[:, 0]) + x
    y_max = np.max(rotated_vertices[:, 1]) + y
    z_max = np.max(rotated_vertices[:, 2]) + z
    box = np.array([[x_min, y_min, z_min, x_max, y_max, z_max]])
    return box

def compute_ciou_loss(box1, box2 , i):
    # Calculate IoU
    iou = my_iou_3d(box1, box2)
    #
    output_box1 = box1.tolist()
    output_box2 = box2.tolist()
    center_dist = np.sqrt((output_box1[3] - output_box2[3]) * (output_box1[3] - output_box2[3]) + (output_box1[4] - output_box2[4]) * (output_box1[4] - output_box2[4]) +(output_box1[5] - output_box2[5]) * (output_box1[5] - output_box2[5]))
    
    box1 = com_box(output_box1[0],output_box1[1],output_box1[2],output_box1[3],output_box1[4],output_box1[5],output_box1[6])
    box2 = com_box(output_box2[0],output_box2[1],output_box2[2],output_box2[3],output_box2[4],output_box2[5],output_box2[6])

    # 拼接两个box并找到外接3D框
    boxes = np.concatenate([box1, box2], axis=0)
    min_xyz = np.min(boxes[:, :3], axis=0)
    max_xyz = np.max(boxes[:, 3:], axis=0)
    bounding_box = np.concatenate([min_xyz, max_xyz], axis=0)

    bound_length = np.sqrt((np.linalg.norm(bounding_box[3:] - bounding_box[:3])) * (np.linalg.norm(bounding_box[3:] - bounding_box[:3])))
    
    # Calculate aspect ratio term
    v = 4 / (np.pi ** 2) * (np.square(np.arctan(output_box1[0] / output_box1[1]) - np.arctan(output_box2[0] / output_box2[1])) + np.square(np.arctan(output_box1[0] / output_box1[2]) - np.arctan(output_box2[0] / output_box2[2])) +np.square(np.arctan(output_box1[1] / output_box1[2]) - np.arctan(output_box2[1] / output_box2[2])))

    a = v /((1-iou+1e-7)+v)
    v1 = a*v
    # Calculate convexity term
    v2 = (center_dist / bound_length) ** 2
    # if (i == 63):
    #     # print("output_box1",output_box1)
    #     # print("output_box2", output_box2)
    #     print("iou_____________________________:",iou)
    #     # print("bound_length",bound_length)
    #     # Calculate center distance
    #     # print("center_dist:",center_dist)
    #     # print("v1",v1)
    #     # print("v2",v2)
    # Calculate CIOU Loss
    ciou_loss = 1 - iou + v1 + v2

    return ciou_loss

class M2TRACK(base_model.MotionBaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.seg_acc = Accuracy(num_classes=2, average='none')
        # print("M2TRACKM2TRACKM2TRACKM2TRACKM2TRACKM2TRACK")
        # print("M2TRACKM2TRACKM2TRACKM2TRACKM2TRACKM2TRACK")
        # print("M2TRACKM2TRACKM2TRACKM2TRACKM2TRACKM2TRACK")
        # print("M2TRACKM2TRACKM2TRACKM2TRACKM2TRACKM2TRACK")
        # print("M2TRACKM2TRACKM2TRACKM2TRACKM2TRACKM2TRACK")
        # print("M2TRACKM2TRACKM2TRACKM2TRACKM2TRACKM2TRACK")
        # print("M2TRACKM2TRACKM2TRACKM2TRACKM2TRACKM2TRACK")
        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_second_stage = getattr(config, 'use_second_stage', True)
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', True)
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)
        if self.use_second_stage:
            self.mini_pointnet2 = MiniPointNet(input_channel=3 + (9 if self.box_aware else 0),
                                               per_point_mlp=[64, 128, 256, 512],
                                               hidden_mlp=[512, 256],
                                               output_size=-1)

            self.box_mlp = nn.Sequential(nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 4))
        if self.use_prev_refinement:
            self.final_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))
        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = Accuracy(num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)

        }

        Returns: B,4

        """
        # print("forwardforwardforwardforwardforwardforward")
        # print("forwardforwardforwardforwardforwardforward")
        # print("forwardforwardforwardforwardforwardforward")
        # print("forwardforwardforwardforwardforwardforward")
        # print("forwardforwardforwardforwardforwardforward")
        # print("forwardforwardforwardforwardforwardforward")
        # print("forwardforwardforwardforwardforwardforward")
        # print("forwardforwardforwardforwardforwardforward")
        output_dict = {}
        x = input_dict["points"].transpose(1, 2)
        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2)
            x = torch.cat([x, candidate_bc], dim=1)

        B, _, N = x.shape

        seg_out = self.seg_pointnet(x)
        seg_logits = seg_out[:, :2, :]  # B,2,N
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls
        mask_xyz_t0 = mask_points[:, :3, :N // 2]  # B,3,N//2
        mask_xyz_t1 = mask_points[:, :3, N // 2:]
        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            # mask_pred_bc_t0 = mask_pred_bc[:, :, :N // 2]  # B,9,N//2
            # mask_pred_bc_t1 = mask_pred_bc[:, :, N // 2:]
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        point_feature = self.mini_pointnet(mask_points)

        # motion state prediction
        motion_pred = self.motion_mlp(point_feature)  # B,4
        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits
        else:
            motion_pred_masked = motion_pred
        # previous bbox refinement
        if self.use_prev_refinement:
            prev_boxes = self.final_mlp(point_feature)  # previous bb, B,4
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # 2nd stage refinement
        if self.use_second_stage:
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2),
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)  # B,3,N//2
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,N

            # transform to the aux_box coordinate system
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
                                                                       aux_box).transpose(1, 2)

            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)
            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4
            output = points_utils.get_offset_box_tensor(aux_box, output_offset)
            output_dict["estimation_boxes"] = output
        else:
            output_dict["estimation_boxes"] = aux_box
        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': aux_box,
                            })

        return output_dict

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits']
        with torch.no_grad():
            seg_label = data['seg_label']
            box_label = data['box_label']
            box_label_prev = data['box_label_prev']
            motion_label = data['motion_label']
            motion_state_label = data['motion_state_label']
            bbox_wlh_label  = data['bbox_size']
            center_label = box_label[:, :3]
            angle_label = torch.sin(box_label[:, 3])
            center_label_prev = box_label_prev[:, :3]
            angle_label_prev = torch.sin(box_label_prev[:, 3])
            center_label_motion = motion_label[:, :3]
            angle_label_motion = torch.sin(motion_label[:, 3])

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
            # print("loss_total__use_motion_cls:",loss_total)
#####################################################################################################
            ciou_loss_sum = 0.0  # 存储所有样本的ciou损失之和
            batch_size = 64
            center_label_motion_numpy = center_label_motion.cpu().numpy()
            angle_label_motion_numpy = angle_label_motion.cpu().numpy()
            center_motion_pred_numpy = motion_pred[:, :3].detach().cpu().numpy()
            angle_motion_pred_numpy = torch.sin(motion_pred[:, 3]).detach().cpu().numpy()
            bbox_wlh_label_numpy = bbox_wlh_label.cpu().numpy()
            # print("bbox_wlh_label_numpy:",bbox_wlh_label_numpy)
            # print("center_label_motion_numpy : ",center_label_motion_numpy)
            # print("center_label_motion_numpy[0,1]",center_label_motion_numpy[0,1])
            for i in range(batch_size):
                x_gt = center_label_motion_numpy[i,0]
                y_gt = center_label_motion_numpy[i,1]
                z_gt = center_label_motion_numpy[i,2]
                angle_gt = angle_label_motion_numpy[i]
                
                width =  bbox_wlh_label_numpy[i,0]
                length =  bbox_wlh_label_numpy[i,1]
                height =  bbox_wlh_label_numpy[i,2]

                x_pred = center_motion_pred_numpy[i,0]
                y_pred = center_motion_pred_numpy[i,1]
                z_pred = center_motion_pred_numpy[i,2]
                angle_pred = angle_motion_pred_numpy[i]

                box_gt = np.array([length,width,height,x_gt,y_gt,z_gt,angle_gt])
                box_pred = np.array([length,width,height,x_pred,y_pred,z_pred,angle_pred])

                ciou_loss = compute_ciou_loss(box_gt,box_pred,i)
                ciou_loss_sum += ciou_loss
                ciou_loss_avg = ciou_loss_sum / batch_size
                # ciou_loss_avg_weight = ciou_loss_avg * 1.5
            ciou_loss_motion = torch.tensor(ciou_loss_avg, device='cuda')
            # print("ciou_loss_motion :",ciou_loss_motion)
            # loss_total += 1 * ciou_loss_motion
            # loss_total += ciou_loss_motion.item()
            # print("loss_total_ciou_loss_use_motion_cls:",loss_total)
#####################################################################################################

        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)

        if self.use_second_stage:
            estimation_boxes = output['estimation_boxes']  # B,4
            loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
            loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
            

#####################################################################################################
            # ciou_loss_sum = 0.0  # 存储所有样本的ciou损失之和
            # batch_size = 64
            # center_label_numpy = center_label.cpu().numpy()
            # angle_label_numpy = angle_label_motion.cpu().numpy()
            # center_pred_numpy = estimation_boxes[:, :3].detach().cpu().numpy()
            # angle_pred_numpy = torch.sin(estimation_boxes[:, 3]).detach().cpu().numpy()
            # bbox_wlh_label_numpy = bbox_wlh_label.cpu().numpy()
            # # print("bbox_wlh_label_numpy:",bbox_wlh_label_numpy)
            # # print("center_label_numpy : ",center_label_numpy)
            # # print("center_label_numpy[0,1]",center_label_numpy[0,1])
            # for i in range(batch_size):
            #     x_gt = center_label_numpy[i,0]
            #     y_gt = center_label_numpy[i,1]
            #     z_gt = center_label_numpy[i,2]
            #     angle_gt = angle_label_numpy[i]
                
            #     width =  bbox_wlh_label_numpy[i,0]
            #     length =  bbox_wlh_label_numpy[i,1]
            #     height =  bbox_wlh_label_numpy[i,2]

            #     x_pred = center_pred_numpy[i,0]
            #     y_pred = center_pred_numpy[i,1]
            #     z_pred = center_pred_numpy[i,2]
            #     angle_pred = angle_pred_numpy[i]

            #     box_gt = np.array([length,width,height,x_gt,y_gt,z_gt,angle_gt])
            #     box_pred = np.array([length,width,height,x_pred,y_pred,z_pred,angle_pred])

            #     ciou_loss = compute_ciou_loss(box_gt,box_pred,i)
            #     ciou_loss_sum += ciou_loss
            #     ciou_loss_avg = ciou_loss_sum / batch_size
            #     # ciou_loss_avg_weight = ciou_loss_avg * 1.5
            # ciou_loss_motion = torch.tensor(ciou_loss_avg, device='cuda')
            # # print("ciou_loss_motion :",ciou_loss_motion)
            # # loss_total += 1 * ciou_loss_motion
            # # loss_total += ciou_loss_motion.item()
            # print("loss_total_ciou_loss_use_second_stage:",loss_total)
#####################################################################################################
            loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
            loss_dict["loss_center"] = loss_center
            loss_dict["loss_angle"] = loss_angle
            # print("loss_total__use_second_stage:",loss_total)
        if self.use_prev_refinement:
            estimation_boxes_prev = output['estimation_boxes_prev']  # B,4
            loss_center_prev = F.smooth_l1_loss(estimation_boxes_prev[:, :3], center_label_prev)
            loss_angle_prev = F.smooth_l1_loss(torch.sin(estimation_boxes_prev[:, 3]), angle_label_prev)

#####################################################################################################
            ciou_loss_sum = 0.0  # 存储所有样本的ciou损失之和
            batch_size = 64
            center_label_prev_numpy = center_label_prev.cpu().numpy()
            angle_label_prev_numpy = angle_label_prev.cpu().numpy()
            center_prev_pred_numpy = estimation_boxes_prev[:, :3].detach().cpu().numpy()
            angle_prev_pred_numpy = torch.sin(estimation_boxes_prev[:, 3]).detach().cpu().numpy()
            bbox_wlh_label_numpy = bbox_wlh_label.cpu().numpy()
            # print("bbox_wlh_label_numpy:",bbox_wlh_label_numpy)
            # print("center_label_numpy : ",center_label_numpy)
            # print("center_label_numpy[0,1]",center_label_numpy[0,1])
            for i in range(batch_size):
                x_gt = center_label_prev_numpy[i,0]
                y_gt = center_label_prev_numpy[i,1]
                z_gt = center_label_prev_numpy[i,2]
                angle_gt = angle_label_prev_numpy[i]
                
                width =  bbox_wlh_label_numpy[i,0]
                length =  bbox_wlh_label_numpy[i,1]
                height =  bbox_wlh_label_numpy[i,2]

                x_pred = center_prev_pred_numpy[i,0]
                y_pred = center_prev_pred_numpy[i,1]
                z_pred = center_prev_pred_numpy[i,2]
                angle_pred = angle_prev_pred_numpy[i]

                box_gt = np.array([length,width,height,x_gt,y_gt,z_gt,angle_gt])
                box_pred = np.array([length,width,height,x_pred,y_pred,z_pred,angle_pred])

                ciou_loss = compute_ciou_loss(box_gt,box_pred,i)
                ciou_loss_sum += ciou_loss
                ciou_loss_avg = ciou_loss_sum / batch_size
                # ciou_loss_avg_weight = ciou_loss_avg * 1.5
            ciou_loss_motion = torch.tensor(ciou_loss_avg, device='cuda')
            # print("ciou_loss_motion :",ciou_loss_motion)
            loss_total += 1 * ciou_loss_motion
            loss_total += ciou_loss_motion.item()
            # print("loss_total_ciou_loss_use_prev_refinement:",loss_total)
#####################################################################################################

            loss_total += (loss_center_prev * self.config.center_weight + loss_angle_prev * self.config.angle_weight)
            loss_dict["loss_center_prev"] = loss_center_prev
            loss_dict["loss_angle_prev"] = loss_angle_prev
            # print("loss_total__use_prev_refinement:",loss_total)   
        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)

        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (
                              loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight)
        # print("loss_total__all:",loss_total)
        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
        })
        if self.box_aware:
            prev_bc = data['prev_bc']
            this_bc = data['this_bc']
            bc_label = torch.cat([prev_bc, this_bc], dim=1)
            pred_bc = output['pred_bc']
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
            # print("loss_total__box_aware:",loss_total)
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        start_time_output = time.time()  # 记录开始时间
        output = self(batch)
        start_time_loss_dict = time.time()  # 记录开始时间
        # print("output = self(batch):" ,start_time_loss_dict - start_time_output)
        # print("output = self(batch):" ,start_time_loss_dict - start_time_output)
        # print("output = self(batch):" ,start_time_loss_dict - start_time_output)
        # print("output = self(batch):" ,start_time_loss_dict - start_time_output)
        # print("output = self(batch):" ,start_time_loss_dict - start_time_output)
        # print("output = self(batch):" ,start_time_loss_dict - start_time_output)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']
        start_time_seg_acc = time.time()  # 记录开始时间 
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # print("loss_dict = self.compute_loss(batch, output):" ,start_time_seg_acc - start_time_loss_dict)
        # log
        seg_acc = self.seg_acc(torch.argmax(output['seg_logits'], dim=1, keepdim=False), batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        start_time_use_motion_cls = time.time()
        # print("seg_acc = self.seg_acc:" ,start_time_use_motion_cls - start_time_seg_acc)
        # print("seg_acc = self.seg_acc:" ,start_time_use_motion_cls - start_time_seg_acc)
        # print("seg_acc = self.seg_acc:" ,start_time_use_motion_cls - start_time_seg_acc)
        # print("seg_acc = self.seg_acc:" ,start_time_use_motion_cls - start_time_seg_acc)
        # print("seg_acc = self.seg_acc:" ,start_time_use_motion_cls - start_time_seg_acc)
        # print("seg_acc = self.seg_acc:" ,start_time_use_motion_cls - start_time_seg_acc)
        # print("seg_acc = self.seg_acc:" ,start_time_use_motion_cls - start_time_seg_acc)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(output['motion_cls'], dim=1, keepdim=False),
                                         batch['motion_state_label'])
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        start_time_selflogger = time.time()
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        # print("start_time_self.logger.experiment.add_scalars:" ,start_time_selflogger - start_time_use_motion_cls)
        return loss


