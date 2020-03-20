import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--point_size', type=int, default=512)
parser.add_argument('--min_octree_threshold', type=float, default=0.04)
parser.add_argument('--max_octree_threshold', type=float, default=0.15)
parser.add_argument('--interval_size', type=float, default=0.035)
parser.add_argument('--scene_path', type=str, default="data/scene_0.h5")
parser.add_argument('--use_vis', type=int, default="1")

opt = parser.parse_args()


from utils.vis_utils import vis_pointcloud,Vis_color
import torch
import time
from GLtree.interval_tree import RedBlackTree, Node, BLACK, RED, NIL
from GLtree.octree_vis_only import point3D
import numpy as np
from utils.ply_tuils import write_ply,create_color_palette,label_mapper
import random
import h5py

point_size = opt.point_size

print("[INFO] load data")

data_file=h5py.File(opt.scene_path,"r")

color_image_array=data_file['color_image']
valid_pose_array=data_file['pose_valid']
points_array=data_file['points_array']
mask_array=data_file['mask']

x_rb_tree = RedBlackTree(opt.interval_size)
y_rb_tree = RedBlackTree(opt.interval_size)
z_rb_tree = RedBlackTree(opt.interval_size)

vis_p=vis_pointcloud(opt.use_vis)
vis_c=Vis_color(opt.use_vis)

frame_index=0
print("[INFO] begin")

with torch.no_grad():
    for i in range(0,color_image_array.shape[0]):
        print("---------------------------")
        print("image:",i)
        time_s=time.time()
        color_image=color_image_array[i,:,:,:].astype(np.uint8)
        points=points_array[i,:,:]
        points_mask=mask_array[i,:,:]
        valid_pose=valid_pose_array[i]
        if valid_pose==0:
            continue

        x_tree_node_list=[]
        y_tree_node_list=[]
        z_tree_node_list=[]
        per_image_node_set=set()

        for p in range(point_size):
        
            x_temp_node = x_rb_tree.add(points[p,0])
            y_temp_node = y_rb_tree.add(points[p,1])
            z_temp_node = z_rb_tree.add(points[p,2])
            x_tree_node_list.append(x_temp_node)
            y_tree_node_list.append(y_temp_node)
            z_tree_node_list.append(z_temp_node)

        for p in range(point_size):

            x_set_union = x_tree_node_list[p].set_list
            y_set_union = y_tree_node_list[p].set_list
            z_set_union = z_tree_node_list[p].set_list
            set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0]
            temp_branch = [None, None, None, None, None, None, None, None]
            temp_branch_distance = np.full((8),opt.max_octree_threshold)
            is_find_nearest = False
            branch_record = set()
            list_intersection=list(set_intersection)
            random.shuffle(list_intersection)

            for point_iter in list_intersection:
                distance = np.sum(np.absolute(point_iter.point_coor - points[p,:]))
                if distance < opt.min_octree_threshold:
                    is_find_nearest = True
                    if frame_index!=point_iter.frame_id:
                        #2D3D fusion
                        point_iter.frame_id=frame_index
                    per_image_node_set.add(point_iter)
                    break
                x = int(point_iter.point_coor[0] >= points[p, 0])
                y = int(point_iter.point_coor[1] >= points[p, 1])
                z = int(point_iter.point_coor[2] >= points[p, 2])
                branch_num= x * 4 + y * 2 + z
                if distance < point_iter.branch_distance[7-branch_num]:
                    branch_record.add((point_iter, 7 - branch_num, distance))
                    if distance < temp_branch_distance[branch_num]:
                        temp_branch[branch_num] = point_iter
                        temp_branch_distance[branch_num] = distance

            if not is_find_nearest:
                new_3dpoint = point3D(points[p, :].T,color_image[int(points_mask[p, 0])*4,
                                    int(points_mask[p, 1])*4,:])
                for point_branch in branch_record:
                    point_branch[0].branch_array[point_branch[1]] = new_3dpoint
                    point_branch[0].branch_distance[point_branch[1]] = point_branch[2]

                new_3dpoint.branch_array = temp_branch
                new_3dpoint.branch_distance = temp_branch_distance
                per_image_node_set.add(new_3dpoint)

                for x_set in x_set_union:
                    x_set.add(new_3dpoint)
                for y_set in y_set_union:
                    y_set.add(new_3dpoint)
                for z_set in z_set_union:
                    z_set.add(new_3dpoint)

        node_lengths=len(per_image_node_set)

        points = np.zeros([node_lengths, 3])
        points_color = np.zeros([node_lengths,3])

        set_count=0
        for set_point in per_image_node_set:
            points[set_count,:]=set_point.point_coor
            points_color[set_count,:]=set_point.point_color
            set_count+=1

        frame_index+=1
        print("time per frame",time.time()-time_s)
        vis_p.update(points,points_color)
        vis_c.update(color_image)

point_result=x_rb_tree.all_points_from_tree(return_color=True)
write_ply(point_result[:,:3],hasrgb=True,rgb_cloud=point_result[:,3:],output_dir="./",name="result_GLtree")

del x_rb_tree
del y_rb_tree
del z_rb_tree
vis_p.run()