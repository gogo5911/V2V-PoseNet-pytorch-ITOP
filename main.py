import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from v2v_util import V2VVoxelization
from v2v_model import V2VModel
from itop import ITOPDataset
from solver import train_epoch, val_epoch, test_epoch

#######################################################################################
## Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='V2V PyTorch ITOP Estimation Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--resume', '-r', default=-1, type=int, help='resume after epoch')
    args = parser.parse_args()
    return args


#######################################################################################
## Configurations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.memory_allocated(0)/1024**3,1
dtype = torch.float

args = parse_args()
resume_train = args.resume >= 0
resume_after_epoch = args.resume

save_checkpoint = True
checkpoint_per_epochs = 1
checkpoint_dir = r'./checkpoint'

start_epoch = 0
epochs_num = 140

batch_size = 12


#######################################################################################
#### ITOP Data Set######
print('==> Preparing data ..')
batch_size = 12
keypoints_num = 15

train_data_dir = 'D:/HumanDataset/ITOP_side_train_depth_map.h5'#학습 data 경로
train_data_labels_dir = 'D:/HumanDataset/ITOP_side_train_labels.h5' #center data 경로

test_data_dir = 'D:/HumanDataset/ITOP_side_test_depth_map.h5'
test_data_labels_dir = 'D:/HumanDataset/ITOP_side_test_labels.h5'

# Transform
voxelization_train = V2VVoxelization(cubic_size=2.0, augmentation=False) #Trainsform을 위해서 만들어 놓는 부분
voxelization_val = V2VVoxelization(cubic_size=2.0, augmentation=False)


def transform_train(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))

def transform_val(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))


# Dataset and loader
train_set = ITOPDataset(train_data_dir, train_data_labels_dir,test_data_dir,test_data_labels_dir, 'train', transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# No separate validation dataset, just use test dataset instead
#val_set = ITOPDataset(train_data_dir, train_data_labels_dir,test_data_dir,test_data_labels_dir, 'test', transform_val)
#val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

#######################################################################################
## Model, criterion and optimizer
print('==> Constructing model ..')
net = V2VModel(input_channels=1, output_channels=keypoints_num)

net = net.to(device, dtype)
if device == torch.device('cuda'):
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    print('cudnn.enabled: ', torch.backends.cudnn.enabled)

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters())

#######################################################################################
## Resume
if resume_train:
    # Load checkpoint
    epoch = 39 #현재 진행된 epoch수
    checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')

    print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

#######################################################################################
## Train and Validate

print('==> Training ..')
for epoch in range(start_epoch, start_epoch + epochs_num):
    print('Epoch: {}'.format(epoch))
    train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)
    #val_epoch(net, criterion, val_loader, device=device, dtype=dtype)

    if save_checkpoint and epoch % checkpoint_per_epochs == 0:
        if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_file)


#######################################################################################
## Test
print('==> Testing ..')
voxelize_input = voxelization_train.voxelize
evaluate_keypoints = voxelization_train.evaluate


def transform_test(sample):
    points, refpoint = sample['points'], sample['refpoint']
    input = voxelize_input(points, refpoint)
    return torch.from_numpy(input), torch.from_numpy(refpoint.reshape((1, -1)))


def transform_output(heatmaps, refpoints):
    keypoints = evaluate_keypoints(heatmaps, refpoints)
    return keypoints


class BatchResultCollector():
    def __init__(self, samples_num, transform_output):
        self.samples_num = samples_num
        self.transform_output = transform_output
        self.keypoints = None
        self.idx = 0

    def __call__(self, data_batch):
        inputs_batch, outputs_batch, extra_batch = data_batch
        outputs_batch = outputs_batch.cpu().numpy()
        refpoints_batch = extra_batch.cpu().numpy()

        keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

        if self.keypoints is None:
            # Initialize keypoints until dimensions awailable now
            self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

        batch_size = keypoints_batch.shape[0]
        self.keypoints[self.idx:self.idx+batch_size] = keypoints_batch
        self.idx += batch_size

    def get_result(self):
        return self.keypoints


print('Test on test dataset ..')
def save_keypoints(filename, keypoints):
    # 하나의 샘플 키포인트를 한줄로 모양 변경
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')


test_set = ITOPDataset(train_data_dir, train_data_labels_dir,test_data_dir,test_data_labels_dir, 'test', transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_res_collector = BatchResultCollector(len(test_set), transform_output)

test_epoch(net, test_loader, test_res_collector, device, dtype)
keypoints_test = test_res_collector.get_result()
save_keypoints('./ITOP_test_res.txt', keypoints_test)


print('Fit on train dataset ..')
fit_set = ITOPDataset(train_data_dir, train_data_labels_dir,test_data_dir,test_data_labels_dir, 'train', transform_test)
fit_loader = torch.utils.data.DataLoader(fit_set, batch_size=batch_size, shuffle=False, num_workers=0)
fit_res_collector = BatchResultCollector(len(fit_set), transform_output)

test_epoch(net, fit_loader, fit_res_collector, device, dtype)
keypoints_fit = fit_res_collector.get_result()
save_keypoints('./fit_res.txt', keypoints_fit)

print('All done ..')