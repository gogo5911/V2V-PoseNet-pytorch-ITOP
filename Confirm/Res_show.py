import h5py
import numpy as np
import matplotlib.pyplot as plt


def world2pixel(x,y,z) :

    w_x = 160 + x / (0.0035 * z)
    w_y = 120 - y /(0.0035 * z)
    return w_x, w_y

def divide_list(l, n):
    # 리스트 l의 길이가 n이면 계속 반복
    for i in range(0, len(l), n):
        yield l[i:i + n]

f = open(".././res/ITOP_test_res.txt", 'r')
result_list = []
joint_num = 15
while True:
    line = f.readline()
    line_split = line.split()
    n = 3
    result = list(divide_list(line_split, n))
    result_list.append(result)
    if not line: break
f.close()

T = h5py.File('D:/HumanDataset/ITOP_side_test_depth_map.h5', 'r') #ITOP 데이터셋 depthmap 가져오는 부분
data, ids = T.get('data'), T.get('id') #depth map과 아이디를 구분해서 넣는 부분
data, ids = np.asarray(data), np.asarray(ids) #array로 변경하는 부분
depthmap = np.asarray(data) #array로 변경하는 부분

res_len = len(result_list)

for fid in range(res_len):
    fx,fy = [],[]
    for i in range(joint_num):
        fx_t, fy_t = world2pixel(float(result_list[fid][i][0]),float(result_list[fid][i][1]), float(result_list[fid][i][2]))
        fx.append(fx_t)
        fy.append(fy_t)


    mfx = max(fx)
    mfy = max(fy)

    if mfx <= 320 and mfy <=240:
        dep = np.array(depthmap[fid], dtype=np.float32)
        plt.scatter(fx, fy, s = 100, c = '#1f77b4')
        plt.imshow(dep)
        plt.show()
        #plt.savefig('.././figs/savefig_default' + str(fid) +'.png')
