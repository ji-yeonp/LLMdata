import pickle
import time
from utils.utils import *
from progress.bar import Bar
import cv2

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.optitrack_dataset import OptiTrackDataset
from network.cross_multi_st_former_3_3 import MultiModalTransformer
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
num_heads = 4
mlp_drop = 0.1

PRETRAIN_FOLDER = 'experiments-qa'
# Antifragile -> 2023-06-25-16-53-10
# Attnetion -> 2023-06-26-04-48-22
# PinkVenom -> 2023-06-26-13-46-43
# Shutdown -> 2023-06-26-19-18-49
time_stamp = '2023-06-26-19-18-49'
repNo = '001'
pretrained = os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, 'best-model-params.pth.tar')
2
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
        music_channels=music_channels,
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
        batch_size=batch_size, num_workers=1,
    )

    loss_qa = AverageMeter()
    batch_time = AverageMeter()

    model.eval()

    bar = Bar('Test', max=len(valid_dataloader))
    # ke_list = list()
    input_list = list()
    score_list = list()
    label_list = list()
    name_list = list()
    music_list = list()
    # spat_att_list_1 = {f'depth_{d}': list() for d in range(depth // 2)}
    # spat_att_list_2 = {f'depth_{d}': list() for d in range(depth // 2)}
    spat_f_att_list = {f'depth_{d}': list() for d in range(depth // 2)}
    temp_f_att_list = {f'depth_{d}': list() for d in range(depth)}
    for ii, (joints, bones, motions, music, labels, names) in enumerate(valid_dataloader):
        input_list.append(joints.numpy())
        music_list.append(music.numpy())
        start_time = time.time()

        inputs_1 = bones.to(device).float()
        inputs_2 = motions.to(device).float()
        music = music.to(device).float()
        labels = labels.to(device).float()

        with torch.no_grad():
            outputs_1, outputs_2 = model(inputs_1, inputs_2, music)
        outputs = (outputs_1 + outputs_2) / 2
        loss = criterion(outputs_1, labels) + criterion(outputs_2, labels)

        loss_qa.update(loss.item(), n=1)
        score_list.extend(outputs.cpu().detach().numpy())
        label_list.extend(labels.cpu().detach().numpy())
        name_list.extend(names)

        for d in range(depth):
            # spat_att_list_1[f'depth_{d}'].append(
            #     np.reshape(model.encoder.spatial_blocks_1[d].attn.attn.cpu().detach().numpy(),
            #                (N, num_tokens, 1, num_joints, num_joints))
            # )
            # spat_att_list_2[f'depth_{d}'].append(
            #     np.reshape(model.encoder.spatial_blocks_2[d].attn.attn.cpu().detach().numpy(),
            #                (N, num_tokens, 1, num_joints, num_joints))
            # )

            if d < depth // 2:
                spat_f_att_list[f'depth_{d}'].append(
                    np.reshape(model.encoder.spatial_blocks_f[d].attn.attn.cpu().detach().numpy(),
                               (-1, num_tokens, 1, 2 * num_joints, 2 * num_joints))
                )
            temp_f_att_list[f'depth_{d}'].append(model.encoder.cross_blocks[d].attn.attn.cpu().detach().numpy())

        batch_time.update(time.time() - start_time)

        bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss_q}'\
            .format(batch=ii + 1, size=len(valid_dataloader), bt=batch_time.avg, eta=bar.eta_td, loss_q=loss_qa.avg)
        bar.next()
    bar.finish()

    # ke_list = np.asarray(ke_list)
    input_list = np.concatenate(input_list, axis=0)
    music_list = np.concatenate(music_list, axis=0)
    score_list = np.asarray(score_list)
    label_list = np.asarray(label_list)

    print(input_list.shape)

    # print(ke_list.shape)
    # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'kinematic_entropy.npy'), ke_list)
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'good-joint.npy'), input_list[label_list < 0.0])
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'good-music.npy'), music_list[label_list < 0.0])
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'good-score.npy'), score_list[label_list < 0.0])
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'good-label.npy'), label_list[label_list < 0.0])

    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'poor-joint.npy'), input_list[label_list > 0.0])
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'poor-music.npy'), music_list[label_list > 0.0])
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'poor-score.npy'), score_list[label_list > 0.0])
    np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'poor-label.npy'), label_list[label_list > 0.0])

    # joint_dist = np.zeros((depth, 16))
    for d in range(depth):
        # spat_att_list_1[f'depth_{d}'] = np.concatenate(spat_att_list_1[f'depth_{d}'], axis=0)
        # spat_att_list_2[f'depth_{d}'] = np.concatenate(spat_att_list_2[f'depth_{d}'], axis=0)
        # temp_att_list[f'depth_{d}'] = np.concatenate(temp_att_list[f'depth_{d}'], axis=0)

        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'spatial_attention_{d}.npy'), spat_att_list[f'depth_{d}'])
        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'temporal_attention_{d}.npy'), temp_att_list[f'depth_{d}'])

        if d < depth // 2:
            spat_f_att_list[f'depth_{d}'] = np.concatenate(spat_f_att_list[f'depth_{d}'], axis=0)
        temp_f_att_list[f'depth_{d}'] = np.concatenate(temp_f_att_list[f'depth_{d}'], axis=0)

        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'cross-attention-{d}-poor.npy'), temp_f_att_list[f'depth_{d}'][label_list < -0.3].mean(axis=(0, 1)))
        # np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'cross-attention-{d}-good.npy'), temp_f_att_list[f'depth_{d}'][label_list > +0.3].mean(axis=(0, 1)))

        if d < depth // 2:
            np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'good-spat-att-{d}.npy'), spat_f_att_list[f'depth_{d}'][label_list < 0.0])
            np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'poor-spat-att-{d}.npy'), spat_f_att_list[f'depth_{d}'][label_list > 0.0])
        np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'good-temp-att-{d}.npy'), temp_f_att_list[f'depth_{d}'][label_list < 0.0])
        np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'poor-temp-att-{d}.npy'), temp_f_att_list[f'depth_{d}'][label_list > 0.0])

    score_correlation(score_list, label_list)
    temporal_pooling_correlation(name_list, score_list, label_list)

    batch_time.reset()
    loss_qa.reset()

if __name__ == "__main__":
    test_model()