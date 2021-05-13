import h5py
import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D

def world2pixel(x,y,z) :

    w_x = 160 + x / (0.0035 * z)
    w_y = 120 - y /(0.0035 * z)
    return w_x, w_y

def pixel2world(x, y, z): #2차원에서 3차원으로
    w_x = (x - 160.0) * z * 0.0035
    w_y = (120.0 - y) * z * 0.0035
    w_z = z
    return w_x, w_y, w_z

def depthmap2points(image): #Depthmap을 이용하여 그리드 포인트 생성  / 카메라의 fx,fy은 초점거리
    h, w = image.shape #image.shape를 이용해서 높이, 너비, 채널의 값을 확인 할 수 있다.
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image)
    return points


def discretize(coord, cropped_size):
    '''[-1, 1] -> [0, cropped_size]''' #cropped_size는 88
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord - min_normalized) / scale #cropped size의 큐브에서로 변경해주는 듯



def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes #?,88,96

    # points shape: (n, 3)
    coord = points #points <- depth map

    # note, will consider points within range [refpoint-cubic_size/2, refpoint+cubic_size/2] as candidates

    # normalize(정규화)
    #(정규화하고자 하는 값 - 데이터 값들 중 최소값) / (데이터 값들 중 최대값 - 데이터 값들 중 최소값)
    #
    coord = (coord - refpoint) / (cubic_size/2)  # -> [-1, 1] #중앙이 0으로 -1에서 1이 되는 값으로 변경한다.

    # discretize
    coord = discretize(coord, cropped_size)  # -> [0, cropped_size]
    coord += (original_size / 2 - cropped_size / 2)  # move center to original volume # 중앙을 원래 original로 옮기다.

    # resize around original volume center #원래 볼륨 중심에서 크기 조정
    resize_scale = new_size / 100
    if new_size < 100:
        coord = coord * resize_scale + original_size/2 * (1 - resize_scale)
    elif new_size > 100:
        coord = coord * resize_scale - original_size/2 * (resize_scale - 1)
    else:
        # new_size = 100 if it is in test mode
        pass

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:,0] -= original_size / 2
        original_coord[:,1] -= original_size / 2
        coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
        coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle)
        coord[:,0] += original_size / 2
        coord[:,1] += original_size / 2

    # translation
    # Note, if trans = (original_size/2 - cropped_size/2), the following translation will
    # cancel the above translation(after discretion). It will be set it when in test mode.
    coord -= trans

    return coord


def scattering(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    coord = coord.astype(np.int32) #int형으로 변경

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap
    cubic[coord[:, 0], coord[:, 1], coord[:, 2]] = 1

    return cubic

def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes # Cropped size : 88
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord, cropped_size)

    return cubic

def generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
    _, cropped_size, _ = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord /= pool_factor  # [0, cropped_size/pool_factor]

    # heatmap generation
    output_size = int(cropped_size / pool_factor)
    heatmap = np.zeros((keypoints.shape[0], output_size, output_size, output_size))

    # use center of cell
    center_offset = 0.5

    for i in range(coord.shape[0]):
        xi, yi, zi= coord[i]
        heatmap[i] = np.exp(-(np.power((d3output_x+center_offset-xi)/std, 2)/2 + \
                              np.power((d3output_y+center_offset-yi)/std, 2)/2 + \
                              np.power((d3output_z+center_offset-zi)/std, 2)/2))

    return heatmap



joint_num = 15

f = h5py.File('D:/HumanDataset/ITOP_side_train_depth_map.h5', 'r') #ITOP 데이터셋 depthmap 가져오는 부분
l = h5py.File('D:/HumanDataset/ITOP_side_train_labels.h5', 'r') #ITOP 데이터셋 label 가져오는 부분

data, ids = f.get('data'), f.get('id') #depth map과 아이디를 구분해서 넣는 부분

sg = l.get('segmentation') #segmentation 가져오는 부분
js = l.get('real_world_coordinates') #실제좌표(m)에서 각 body part의 위치에 해당하는 3차원 좌표

data, ids,sga = np.asarray(data), np.asarray(ids),np.asarray(sg) #array로 변경하는 부분

mask = sga >= 0
s = np.where(mask, data, 0) #segmentation 있는 부분만 제거 해서 사용하기 위해 변경하는 부분
depthmap = np.asarray(s) #array로 변경하는 부분


#depthmap = np.asarray(data) #array로 변경하는 부분

joints_world = js[:,:,:]
frameNum = len(joints_world)


for fid in range(frameNum):
    centers = []
    center_x = (joints_world[fid,:,0].max() + joints_world[fid,:,0].min()) / 2
    center_y = (joints_world[fid,:,1].max() + joints_world[fid,:,1].min()) / 2
    center_z = (joints_world[fid,:,2].max() + joints_world[fid,:,2].min()) / 2
    centers.append([center_x, center_y, center_z])

    if center_x == 0 and center_y == 0 and center_z == 0:
        continue

    points = depthmap2points(depthmap[fid]) #Depthmap을 이용하여 그리드 포인트 생성 부분
    points = points.reshape((-1, 3)) #(240,320,3)->(76800,3)으로 변경


    cropped_size, original_size = 88, 96 #cropped_size로 input이 만들어진다.
    sizes = (2, cropped_size, original_size)
    augmentation = False

    ## Augmentations
    # Resize
    new_size = np.random.rand() * 40 + 80

    # Rotation
    angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi

    # Translation
    trans = np.random.rand(3) * (original_size-cropped_size)

    if not augmentation:
        new_size = 100
        angle = 0
        trans = 0

    input = generate_cubic_input(points, centers, new_size, angle, trans, sizes)

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.voxels(input, facecolors='#7D84A6', edgecolors='#7D84A6', antialiased=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=60, azim =30)

    plt.show()
    #plt.savefig('.././figs/savefig_default' + str(fid) +'.png')

