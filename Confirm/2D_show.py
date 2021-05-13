import h5py
import numpy as np
import matplotlib.pyplot as plt

def world2pixel(x,y,z) :

    w_x = 160 + x / (0.0035 * z)
    w_y = 120 - y /(0.0035 * z)
    return w_x, w_y



joint_num = 15

f = h5py.File('D:/HumanDataset/ITOP_side_train_depth_map.h5', 'r') #ITOP 데이터셋 depthmap 가져오는 부분
l = h5py.File('D:/HumanDataset/ITOP_side_train_labels.h5', 'r') #ITOP 데이터셋 label 가져오는 부분

data, ids = f.get('data'), f.get('id') #depth map과 아이디를 구분해서 넣는 부분

sg = l.get('segmentation') #segmentation 가져오는 부분
js = l.get('real_world_coordinates') #실제좌표(m)에서 각 body part의 위치에 해당하는 3차원 좌표

data, ids,sga = np.asarray(data), np.asarray(ids),np.asarray(sg) #array로 변경하는 부분
depthmap = np.asarray(data) #array로 변경하는 부분


joints_world = js[:,:,:]

jw_len = len(joints_world)

for fid in range(jw_len):

    #관절 값을 이용해서 사람의 정중앙 찾기
    center_x = (joints_world[fid,:,0].max() + joints_world[fid,:,0].min()) / 2
    center_y = (joints_world[fid,:,1].max() + joints_world[fid,:,1].min()) / 2
    center_z = (joints_world[fid,:,2].max() + joints_world[fid,:,2].min()) / 2

    if center_x == 0 and center_y == 0 and center_z == 0:
        continue

    fx,fy = [],[]
    for i in range(joint_num):
        fx_t, fy_t = world2pixel(joints_world[fid,i,0],joints_world[fid,i,1],joints_world[fid,i,2])
        fx.append(fx_t)
        fy.append(fy_t)

    cx, cy = world2pixel(center_x, center_y, center_z)
    dep = np.array(depthmap[fid], dtype=np.float32)
    plt.scatter(cx, cy, s = 100, c = 'r')
    plt.scatter(fx, fy, s = 100, c = '#1f77b4')
    plt.imshow(dep)
    plt.show()
    #plt.savefig('.././figs/savefig_default' + str(fid) +'.png')