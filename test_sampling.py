import pickle
import time
from utils.utils import *
from progress.bar import Bar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.optitrack_dataset_all_0 import OptiTrackDataset
from network_mae.temporal_former_0_0 import *
from utils.metric import *


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = 1

STRIDE = 1

NUM_REPETITIONS = 1
num_epochs = 30
window_frames = 256
num_tokens = 64
num_joints = 21
motion_channels = 6
music_channels = 35

embed_dim = 4
depth = 4
num_heads = 1
decoder_embed_dim = 4
decoder_depth = 4
decoder_num_heads = 1
mlp_drop = 0.0

NUM_SPLITS = 2
EXP_FOLDER = 'experiments-qa'
PRETRAIN_FOLDER = 'experiments-mae'
# time_stamp = '2023-08-11-22-33-34' # joint 4-4-1
# time_stamp = '2023-08-14-20-27-18' # joint 8-4-2
# time_stamp = '2023-08-12-12-02-43' # motion difference
time_stamp = '2023-08-12-13-43-21' # motion dependent
# time_stamp = '2023-08-12-22-22-20' # bone dependent
exp_epoch = 'best-model-params.pth.tar'
pretrained = os.path.join(PRETRAIN_FOLDER, time_stamp, exp_epoch)

batch_size = 2048 * NUM_GPUS
mask_ratio = 0.0

# Antifragile, Attention, PinkVenom, Shutdown
# TRAIN_SET = ['Antifragile', 'Attention']
# UPDATE_SET = ['PinkVenom']
# VALID_SET = ['Shutdown']
VALID_SET = ['Antifragile', 'Attention', 'PinkVenom', 'Shutdown']

def test_model():
    model = MaskedAutoencoder(
        window_len=window_frames,
        num_tokens=num_tokens,
        num_joints=num_joints,
        in_channels=motion_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_drop=mlp_drop,
    )
    print(model)

    if pretrained is None:
        print('Training from scratch...')
    else:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(pretrained))
        model.encoder.load_state_dict(checkpoint['encoder_params'])
        model.decoder.load_state_dict(checkpoint['decoder_params'])

    criterion = nn.MSELoss()
    print(criterion)

    print('Total params: %.8fM' % (sum(p.numel() for p in model.parameters()) / 1.0e+6))
    model.to(device)
    criterion.to(device)

    # train_dataloader = DataLoader(
    #     OptiTrackDataset(
    #         dataset_path=[f'./data/{name}' for name in TRAIN_SET],
    #         split='train',
    #         window_frames=window_frames,
    #         max_stride=STRIDE,
    #     ),
    #     batch_size=batch_size, num_workers=4 * NUM_GPUS,
    # )
    # update_dataloader = DataLoader(
    #     OptiTrackDataset(
    #         dataset_path=[f'./data/{name}' for name in UPDATE_SET],
    #         split='update',
    #         window_frames=window_frames,
    #         max_stride=STRIDE,
    #     ),
    #     batch_size=batch_size, num_workers=4,
    # )
    valid_dataloader = DataLoader(
        OptiTrackDataset(
            dataset_path=[f'./data/{name}' for name in VALID_SET],
            split='valid',
            window_frames=window_frames,
            max_stride=STRIDE,
        ),
        batch_size=batch_size, num_workers=4,
    )

    loss_mae = AverageMeter()
    total_mpjpe = AverageMeter()
    batch_time = AverageMeter()

    model.eval()

    data_loaders = {
        # 'train': train_dataloader,
        # 'update': update_dataloader,
        'valid': valid_dataloader,
    }
    for phase, dl in data_loaders.items():
        bar = Bar(phase, max=len(dl))
        feat_list = list()
        output_list = list()
        label_list = list()
        name_list = list()
        frm_index_list = list()
        for ii, (joints, bones, motions, labels, names, frm_index) in enumerate(data_loaders[phase]):
            start_time = time.time()

            inputs = motions.to(device).float()
            labels = labels.to(device).float()

            with torch.no_grad():
                feats, outputs = model(inputs, mask_ratio=mask_ratio)
            loss = criterion(outputs, inputs)

            mpjpe = torch.norm(inputs - outputs, p='fro', dim=-1)

            loss_mae.update(loss.item(), n=1)
            total_mpjpe.update(mpjpe.mean().item(), n=inputs.size(0))
            # output_list.extend(outputs.cpu().detach().numpy())
            # label_list.extend(labels.cpu().detach().numpy())
            feat_list.extend(feats.cpu().detach().numpy())
            name_list.extend(names)
            frm_index_list.extend(frm_index)

            batch_time.update(time.time() - start_time)

            bar.suffix = f'({ii + 1}/{len(valid_dataloader)}) | Batch: {batch_time.avg:.3f}s | ETA: {bar.eta_td:} | Loss: {loss_mae.avg}  | MPJPE: {1000.0 * total_mpjpe.avg:.2f} mm'
            bar.next()
        bar.finish()

        batch_time.reset()
        loss_mae.reset()

        feat_list = np.asarray(feat_list)
        frm_index_list = np.asarray(frm_index_list)

        print(feat_list.shape, frm_index_list.shape)

        get_information_entropy(name_list, frm_index_list, feat_list, f'entropy-str{STRIDE}')
        # get_semantic_distance(name_list, frm_index_list, feat_list, f'feat-dist-str{STRIDE}')

        # feat_dict = get_sequence_dict(name_list, frm_index_list, feat_list, f'feat_str_{STRIDE}')
        # dist_dict = get_distance_dict(feat_dict, f'cdf-str-{STRIDE}')

if __name__ == "__main__":
    test_model()