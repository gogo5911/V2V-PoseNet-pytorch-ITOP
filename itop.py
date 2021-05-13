import numpy as np
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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


class ITOPDataset(Dataset):
    def __init__(self, train_root, train_label_root, test_root, test_label_root, mode,  transform = None):
        self.trainSz = 39795 #학습 수
        self.testSz = 10501 #Test 수
        self.img_widt = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
        self.joint_num = 15
        self.world_dim = 3

        self.train_root = train_root #Train 학습 데이터 경로
        self.t_label_root = train_label_root #Train labels 데이터 경로

        self.test_root = test_root #Test 학습 데이터 경로
        self.te_label_root = test_label_root #Test labels 데이터 경로

        self.transform = transform
        self.mode = mode

        if not self.mode in ['train', 'test']: raise ValueError('모드를 설정해주세요.')

        if self.mode == "train":
            self.frameNum = self.trainSz
            print("training data loading..")
        elif self.mode == "test" :
            self.frameNum = self.testSz
            print("test data loading..")

        self._load()

    def __len__(self):
        return self.frameNum

    def _load(self):


        if self.mode == 'train':
            f = h5py.File(self.train_root, 'r') #ITOP 데이터셋 depthmap 가져오는 부분
            l = h5py.File(self.t_label_root, 'r') #ITOP 데이터셋 label 가져오는 부분
        else:
            f = h5py.File(self.test_root, 'r')
            l = h5py.File(self.te_label_root, 'r')

        data, ids = f.get('data'), f.get('id') #depth map과 아이디를 구분해서 넣는 부분
        js = l.get('real_world_coordinates') #실제좌표(m)에서 각 body part의 위치에 해당하는 3차원 좌표
        sg = l.get('segmentation') #segmentation 가져오는 부분

        #head(1), neck(2), r_shoulder(3), l_shoulder(4), r_elbow(5), l_elbow(6), r_hand(7), l_hand(8)
        #torso(9), r_hip(10), l_hip(11), r_knee(12), l_knee(13), r_foot(14), l_foot(15)



        self.jointWorld = np.zeros((self.frameNum,self.joint_num,self.world_dim), dtype=np.float64)
        self.jointWorld = js[:,:,:] #관절
        self.refpt = np.zeros((self.frameNum, self.world_dim),dtype=np.float64) #중앙점

        centers = []
        for fid in range(self.frameNum):
            self.refpt[fid, 0] = (self.jointWorld[fid,:,0].max() + self.jointWorld[fid,:,0].min()) /2
            self.refpt[fid, 1] = (self.jointWorld[fid,:,1].max() + self.jointWorld[fid,:,1].min()) /2
            self.refpt[fid, 2] = (self.jointWorld[fid,:,2].max() + self.jointWorld[fid,:,2].min()) /2

        self.depthmap, self.ids, self.sga = np.asarray(data), np.asarray(ids), np.asarray(sg)


    def __getitem__(self, index):
        #index 문제
        mask = self.sga[index] >= 0
        s = np.where(mask, self.depthmap[index], 0) #segmentation 있는 부분만 제거 해서 사용하기 위해 변경하는 부분
        self.depthmap_mask = np.asarray(s)

        self.points = depthmap2points(self.depthmap_mask)
        self.point = self.points.reshape((-1, 3)) #(240,320,3)->(76800,3)으로 변경

        sample = {
            'name' : self.ids[index],
            'points' : self.point, #depthmap
            'joints' : self.jointWorld[index], #조인트 월드 좌표
            'refpoint' : self.refpt[index]
        }

        if self.transform: sample = self.transform(sample)

        return sample