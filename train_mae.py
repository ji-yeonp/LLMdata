import pickle
import time
from utils.utils import *
from progress.bar import Bar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.optitrack_dataset_all_1 import OptiTrackDataset
# from network.convnet_frame import *
# from network.unimodal_mae_token import *
# from network.unimodal_mae_frame_posemb import *
# from network_mae.st_former_0 import *
from network_mae.temporal_former_0_0 import *
# from network.st_gcn_0 import Model


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = 4
FPS = 360

NUM_REPETITIONS = 1
num_epochs = 1000
window_frames = 256
num_tokens = 64
num_joints = 21
motion_channels = 3
music_channels = 35

embed_dim = 4
depth = 4
num_heads = 1
decoder_embed_dim = 4
decoder_depth = 4
decoder_num_heads = 1
mlp_drop = 0.0

NUM_SPLITS = 2
EXP_FOLDER = 'experiments-mae'
PRETRAIN_FOLDER = 'experiments-mae'
time_stamp = '2023-04-27-22-17-02'
exp_epoch = 'params_epoch-050.pth.tar'
pretrained = None # os.path.join(PRETRAIN_FOLDER, time_stamp, 'model_parameters', exp_epoch)

batch_size = 768 * NUM_GPUS
initial_lr = 1.5e-4 * batch_size / 256
w_decay = 5e-2
lr_decay = 0.95
resume_epoch = 0
mask_ratio = 0.75

# Antifragile, Attention, PinkVenom, Shutdown
TRAIN_SET = ['Antifragile', 'Attention', 'PinkVenom']
VALID_SET = ['Shutdown']

def train_mae(exp_name):
    if not os.path.exists(EXP_FOLDER):
        os.mkdir(EXP_FOLDER)
    if not os.path.exists(os.path.join(EXP_FOLDER, exp_name)):
        os.mkdir(os.path.join(EXP_FOLDER, exp_name))
        
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

    # if pretrained is None:
    #     print('Training from scratch...')
    # else:
    #     checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
    #     print("Initializing weights from: {}...".format(pretrained))
    #     model.encoder.load_state_dict(checkpoint['encoder_dict'])

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), weight_decay=w_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_decay ** epoch)

    print(criterion)
    print(optimizer)

    print('Total params: %.8fM' % (sum(p.numel() for p in model.parameters()) / 1.0e+6))
    model = nn.DataParallel(model)
    model.to(device)
    criterion.to(device)

    
    train_dataloader = DataLoader(
        OptiTrackDataset(
            dataset_path=[f'./data/{name}' for name in TRAIN_SET],
            split='train',
            window_frames=window_frames,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 * NUM_GPUS,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        OptiTrackDataset(
            dataset_path=[f'./data/{name}' for name in VALID_SET],
            split='valid',
            window_frames=window_frames,
        ),
        batch_size=batch_size,
        num_workers=1 * NUM_GPUS,
        drop_last=True,
    )
    trainval_loaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
    }

    loss_mae = AverageMeter()
    total_mpjpe = AverageMeter()
    batch_time = AverageMeter()

    for i in range(resume_epoch, num_epochs):
        curent_lr = scheduler.get_last_lr()[0]
        print('\nEpoch: {epoch}/{num_epochs} LR: {lr:.8f}'.format(epoch=i + 1, num_epochs=num_epochs, lr=curent_lr))

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            start_time = time.time()
            bar = Bar(phase, max=len(trainval_loaders[phase]))
            name_list = list()
            # input_list = list()
            # output_list = list()
            # stride_list = list()
            for ii, (joints, bones, motions, labels, _, names, _) in enumerate(trainval_loaders[phase]):
                inputs = motions.to(device).float()
                labels = labels.to(device).float()

                if phase == 'train':
                    _, outputs = model(inputs, mask_ratio=mask_ratio)
                    loss = criterion(outputs, inputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        _, outputs = model(inputs, mask_ratio=mask_ratio)
                    loss = criterion(outputs, inputs)

                mpjpe = torch.norm(inputs - outputs, p='fro', dim=-1)

                loss_mae.update(loss.item(), n=1)
                total_mpjpe.update(mpjpe.mean().item(), n=inputs.size(0))
                name_list.extend(names)

                batch_time.update(time.time() - start_time)
                start_time = time.time()

                bar.suffix = f'({ii + 1}/{len(trainval_loaders[phase])}) | Batch: {batch_time.avg:.3f}s | ETA: {bar.eta_td:} | Loss: {loss_mae.avg:.6f} | MPJPE: {1000.0 * total_mpjpe.avg:.2f} mm'
                bar.next()
            bar.finish()

            if phase == 'train':
                scheduler.step()
            elif phase == 'valid':
                save_path = os.path.join(EXP_FOLDER, exp_name, 'best-model-params.pth.tar')
                torch.save({
                    'encoder_params': model.module.encoder.state_dict(),
                    'decoder_params': model.module.decoder.state_dict(),
                }, save_path)
                print(f'Save model at {save_path}')

                # input_list = np.concatenate(input_list, axis=0)
                # output_list = np.concatenate(output_list, axis=0)
                # stride_list = np.concatenate(stride_list, axis=0)

                # mse_stride(input_list, output_list, stride_list)

                # del input_list, output_list, stride_list

            batch_time.reset()
            loss_mae.reset()
            total_mpjpe.reset()

if __name__ == "__main__":
    exp_name = '{year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}'.format(
        year=time.localtime().tm_year,
        month=time.localtime().tm_mon,
        day=time.localtime().tm_mday,
        hour=time.localtime().tm_hour,
        minute=time.localtime().tm_min,
        second=time.localtime().tm_sec
    )
    train_mae(exp_name)