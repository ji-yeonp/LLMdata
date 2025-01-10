import pickle
import time
from utils.utils import *
from progress.bar import Bar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.optitrack_dataset import OptiTrackDataset
from network_proposed.multi_st_former_2 import MultiModalTransformer
from utils.metric import *


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

window_frames = 256
num_tokens = 64
num_joints = 21
motion_channels = 6
music_channels = 35
input_embed = 'token'

embed_dim = 4
depth = 4
num_heads = 1
mlp_drop = 0.0

PRETRAIN_FOLDER = 'experiments-qa'
# Antifragile -> 2023-06-27-14-54-35
# Attnetion -> 
# PinkVenom -> 
# Shutdown -> 
time_stamp = '2023-06-23-15-24-44'
repNo = '001'
pretrained = os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, 'best-model-params.pth.tar')

batch_size = 1024

# Antifragile, Attention, PinkVenom, Shutdown
CHOREO_NAME = 'Shutdown'
NUM_SPLITS = 2

def test_model():
    model = MultiModalTransformer(
        window_len=window_frames,
        num_tokens=num_tokens,
        num_joints=num_joints,
        in_channels=motion_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_drop=mlp_drop,
        input_embed=input_embed,
    )
    print(model)

    if pretrained is None:
        print('Training from scratch...')
    else:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(pretrained))
        model.load_state_dict(checkpoint['model_dict'])

    criterion = nn.MSELoss()
    print(criterion)

    print('Total params: %.8fM' % (sum(p.numel() for p in model.parameters()) / 1.0e+6))
    model.to(device)
    criterion.to(device)
    
    valid_dataloader = DataLoader(
        OptiTrackDataset(
            dataset_path=f'./data/{CHOREO_NAME}',
            split='valid',
            num_split=NUM_SPLITS,
            window_frames=window_frames,
        ),
        batch_size=batch_size, num_workers=0,
    )

    loss_qa = AverageMeter()
    batch_time = AverageMeter()

    model.eval()

    bar = Bar('Test', max=len(valid_dataloader))
    # ke_list = list()
    score_list = list()
    label_list = list()
    name_list = list()
    spat_att_list_1 = {f'depth_{d}': list() for d in range(depth // 2)}
    spat_att_list_2 = {f'depth_{d}': list() for d in range(depth // 2)}
    spat_f_att_list = {f'depth_{d}': list() for d in range(depth // 2)}
    temp_f_att_list = {f'depth_{d}': list() for d in range(depth // 2)}
    for ii, (joints, bones, motions, labels, names) in enumerate(valid_dataloader):
        start_time = time.time()

        inputs_1 = bones.to(device).float()
        inputs_2 = motions.to(device).float()
        labels = labels.to(device).float()

        N, _, _, _ = inputs_1.size()

        with torch.no_grad():
            outputs = model(inputs_1, inputs_2)
        loss = criterion(outputs, labels)

        loss_qa.update(loss.item(), n=1)
        score_list.extend(outputs.cpu().detach().numpy())
        label_list.extend(labels.cpu().detach().numpy())
        name_list.extend(names)

        for d in range(depth // 2):
            spat_att_list_1[f'depth_{d}'].append(
                np.reshape(model.encoder.spatial_blocks_1[d].attn.attn.cpu().detach().numpy(),
                           (N, num_tokens, 1, num_joints, num_joints))
            )
            spat_att_list_2[f'depth_{d}'].append(
                np.reshape(model.encoder.spatial_blocks_2[d].attn.attn.cpu().detach().numpy(),
                           (N, num_tokens, 1, num_joints, num_joints))
            )

            spat_f_att_list[f'depth_{d}'].append(
                np.reshape(model.encoder.spatial_blocks_f[d].attn.attn.cpu().detach().numpy(),
                           (N, num_tokens, 1, 2 * num_joints, 2 * num_joints))
            )
            temp_f_att_list[f'depth_{d}'].append(model.encoder.temporal_blocks_f[d].attn.attn.cpu().detach().numpy())

        batch_time.update(time.time() - start_time)

        bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss_q}'\
            .format(batch=ii + 1, size=len(valid_dataloader), bt=batch_time.avg, eta=bar.eta_td, loss_q=loss_qa.avg)
        bar.next()
    bar.finish()

    # ke_list = np.asarray(ke_list)
    score_list = np.asarray(score_list)
    label_list = np.asarray(label_list)

    # print(ke_list.shape)
    # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'kinematic_entropy.npy'), ke_list)
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'score.npy'), score_list)
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'label.npy'), label_list)

    # joint_dist = np.zeros((depth, 16))
    for d in range(depth // 2):
        spat_att_list_1[f'depth_{d}'] = np.concatenate(spat_att_list_1[f'depth_{d}'], axis=0)
        spat_att_list_2[f'depth_{d}'] = np.concatenate(spat_att_list_2[f'depth_{d}'], axis=0)
        # temp_att_list[f'depth_{d}'] = np.concatenate(temp_att_list[f'depth_{d}'], axis=0)

        print(spat_att_list_1[f'depth_{d}'].shape, spat_att_list_2[f'depth_{d}'].shape)
        for v in spat_att_list_1[f'depth_{d}'].mean(axis=(0, 1, 2)).mean(axis=0):
            print(v)
        for v in spat_att_list_2[f'depth_{d}'].mean(axis=(0, 1, 2)).mean(axis=0):
            print(v)

        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'spatial_attention_{d}.npy'), spat_att_list[f'depth_{d}'])
        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'temporal_attention_{d}.npy'), temp_att_list[f'depth_{d}'])

        spat_f_att_list[f'depth_{d}'] = np.concatenate(spat_f_att_list[f'depth_{d}'], axis=0)
        temp_f_att_list[f'depth_{d}'] = np.concatenate(temp_f_att_list[f'depth_{d}'], axis=0)

        print(spat_f_att_list[f'depth_{d}'].shape, temp_f_att_list[f'depth_{d}'].shape)
        for v in spat_f_att_list[f'depth_{d}'].mean(axis=(0, 1, 2)).mean(axis=0):
            print(v)

        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'spatial_f_attention_{d}.npy'), spat_f_att_list[f'depth_{d}'])
        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'temporal_f_attention_{d}.npy'), temp_f_att_list[f'depth_{d}'])

    plcc_w, srcc_w, krcc_w = score_correlation(score_list, label_list)
    plcc_s, srcc_s, krcc_s = temporal_pooling_correlation(name_list, score_list, label_list)

    batch_time.reset()
    loss_qa.reset()

if __name__ == "__main__":
    test_model()