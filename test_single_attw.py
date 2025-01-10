import pickle
import time
from utils.utils import *
from progress.bar import Bar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.multimodal_dataset import MultimodalDataset
# from network.convnet_frame import *
# from network.unimodal_mae_token import *
# from network.unimodal_mae_frame_posemb import *
# from network_analysis.st_former_2 import *
from network_proposed.multi_st_former_1 import MultiModalTransformer
# from network.st_gcn_0 import Model
from utils.metric import *


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_REPETITIONS = 50
num_epochs = 50
window_len = 256
num_tokens = 64
num_joints = 16
motion_channels = 6
music_channels = 35

embed_dim = 8
depth = 4
num_heads = 1
mlp_drop = 0.0

PRETRAIN_FOLDER = 'experiments-qa'
time_stamp = '2023-06-22-15-54-28'
repNo = '001'
pretrained = os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, 'best-model-parms.pth.tar')

batch_size = 1

def test_model():
    model = UnimodalTransformer(
        window_len=window_len,
        num_tokens=num_tokens,
        num_joints=num_joints,
        in_channels=motion_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_drop=mlp_drop,
    )
    print(model)

    if pretrained is None:
        print('Training from scratch...')
    else:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(pretrained))
        model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.MSELoss()
    print(criterion)

    print('Total params: %.8fM' % (sum(p.numel() for p in model.parameters()) / 1.0e+6))
    model.to(device)
    criterion.to(device)

    valid_dataloader = DataLoader(
        MultimodalDataset(dataset_path='data/Homo', split='valid', window_len=window_len, num_tokens=num_tokens),
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
    spat_att_list = {f'depth_{d}': list() for d in range(depth)}
    temp_att_list = {f'depth_{d}': list() for d in range(depth)}
    for ii, (joints, bones, motions, music, labels, names) in enumerate(valid_dataloader):
        start_time = time.time()

        # kinematic_entropy = F34_beat_align_global_norm(joints.numpy().squeeze(), music.numpy().squeeze())
        # ke_list.append(kinematic_entropy)
        
        inputs = bones.to(device).float()
        labels = labels.to(device).float()

        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss_qa.update(loss.item(), n=1)
        score_list.extend(outputs.cpu().detach().numpy())
        label_list.extend(labels.cpu().detach().numpy())
        name_list.extend(names)

        for d in range(depth):
            spat_att_list[f'depth_{d}'].append(model.encoder.spatial_blocks[d].attn.attn.cpu().detach().numpy().squeeze())
            temp_att_list[f'depth_{d}'].append(model.encoder.temporal_blocks[d].attn.attn.cpu().detach().numpy().squeeze())

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
    for d in range(depth):
        spat_att_list[f'depth_{d}'] = np.stack(spat_att_list[f'depth_{d}'], axis=0)
        temp_att_list[f'depth_{d}'] = np.stack(temp_att_list[f'depth_{d}'], axis=0)

        np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'spatial_attention_{d}.npy'), spat_att_list[f'depth_{d}'])
        np.save(os.path.join(PRETRAIN_FOLDER, time_stamp, repNo, f'temporal_attention_{d}.npy'), temp_att_list[f'depth_{d}'])

    #     for n in range(spat_att_list[f'depth_{d}'].shape[0]):
    #         for l in range(spat_att_list[f'depth_{d}'].shape[1]):
    #             joint_dist[d, np.argmax(spat_att_list[f'depth_{d}'][n, l, :, :].mean(axis=0))] += 1

    # for n in range(joint_dist.shape[1]):
    #     print(f'{n + 1:02d}', [f'{joint_dist[d, n] / np.sum(joint_dist[d, :]):0.8f}' for d in range(depth)])

    plcc_w, srcc_w, krcc_w = score_correlation(score_list, label_list)
    plcc_s, srcc_s, krcc_s = temporal_pooling_correlation(name_list, score_list, label_list)

    batch_time.reset()
    loss_qa.reset()

if __name__ == "__main__":
    test_model()