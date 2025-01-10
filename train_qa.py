import pickle
import time
from utils.utils import *
from progress.bar import Bar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.optitrack_dataset import OptiTrackDataset
# from network.convnet_frame import *
# from network.unimodal_mae_token import *
# from network.unimodal_mae_frame_posemb import *
from network.st_former_2 import *
# from network.st_gcn_0 import Model


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = 4

NUM_REPETITIONS = 1
num_epochs = 100
window_frames = 256
num_tokens = 64
num_joints = 5
motion_channels = 3
music_channels = 35

embed_dim = 4
depth = 4
num_heads = 1
mlp_drop = 0.0

NUM_SPLITS = 2
EXP_FOLDER = 'experiments-qa'
PRETRAIN_FOLDER = 'experiments-mae'
time_stamp = '2023-04-27-22-17-02'
exp_epoch = 'params_epoch-050.pth.tar'
pretrained = None # os.path.join(PRETRAIN_FOLDER, time_stamp, 'model_parameters', exp_epoch)

batch_size = 2048
if pretrained is None:
    # train-from-scratch
    initial_lr = 1e-2
    w_decay = 3e-1
else:
    # fine-tune
    initial_lr = 1e-3
    w_decay = 5e-2
lr_decay = 0.99
resume_epoch = 0

# Antifragile, Attention, PinkVenom, Shutdown
CHOREO_NAME = 'Shutdown'

def train_model(exp_name, repNo):
    if not os.path.exists(EXP_FOLDER):
        os.mkdir(EXP_FOLDER)
    if not os.path.exists(os.path.join(EXP_FOLDER, exp_name)):
        os.mkdir(os.path.join(EXP_FOLDER, exp_name))
    if not os.path.exists(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}')):
        os.mkdir(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'))

    # model = MotionConvNetReshape(
    #     window_len=window_len,
    #     num_tokens=num_tokens,
    #     num_joints=num_joints,
    #     in_channels=motion_channels,
    #     embed_dim=embed_dim,
    #     depth=depth,
    # )
    # model = MotionConvNetMeanpool(
    #     window_len=window_len,
    #     num_tokens=num_tokens,
    #     in_channels=motion_channels,
    #     embed_dim=embed_dim,
    #     depth=depth,
    # )
    model = UnimodalTransformer(
        window_len=window_frames,
        num_tokens=num_tokens,
        num_joints=num_joints,
        in_channels=motion_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_drop=mlp_drop,
    )
    # model = Model(
    #     num_joints=num_joints,
    #     in_channels=motion_channels,
    # )
    print(model)

    if pretrained is None:
        print('Training from scratch...')
    else:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(pretrained))
        model.encoder.load_state_dict(checkpoint['encoder_dict'])

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
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
            dataset_path=f'./data/{CHOREO_NAME}',
            split='train',
            num_split=NUM_SPLITS,
            window_frames=window_frames,
        ),
        batch_size=batch_size, shuffle=True, num_workers=4 * NUM_GPUS,
    )
    valid_dataloader = DataLoader(
        OptiTrackDataset(
            dataset_path=f'./data/{CHOREO_NAME}',
            split='valid',
            num_split=NUM_SPLITS,
            window_frames=window_frames,
        ),
        batch_size=batch_size, num_workers=0,
    )
    trainval_loaders = {'train': train_dataloader, 'valid': valid_dataloader}

    loss_qa = AverageMeter()
    batch_time = AverageMeter()
    train_barometer = {
        'loss': list(),
        'PLCC_win': list(),
        'SRCC_win': list(),
        'KRCC_win': list(),
        'PLCC_seq': list(),
        'SRCC_seq': list(),
        'KRCC_seq': list(),
    }
    valid_barometer = {
        'loss': list(),
        'PLCC_win': list(),
        'SRCC_win': list(),
        'KRCC_win': list(),
        'PLCC_seq': list(),
        'SRCC_seq': list(),
        'KRCC_seq': list(),
    }

    for i in range(resume_epoch, num_epochs):
        curent_lr = scheduler.get_last_lr()[0]
        print('\nEpoch: {epoch}/{num_epochs} LR: {lr:.8f}'.format(epoch=i + 1, num_epochs=num_epochs, lr=curent_lr))

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            end = time.time()
            bar = Bar(phase, max=len(trainval_loaders[phase]))
            score_list = list()
            label_list = list()
            name_list = list()
            for ii, (joints, bones, motions, labels, names) in enumerate(trainval_loaders[phase]):
                inputs = joints[:, :, 0:5, :].to(device).float()
                # inputs = joints[:, :, 5:13, :].to(device).float()
                # inputs = joints[:, :, 13:21, :].to(device).float()
                # inputs = joints[:, :, 5:21, :].to(device).float()
                # inputs = joints[:, :, [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20], :].to(device).float()
                labels = labels.to(device).float()

                if phase == 'train':
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss_qa.update(loss.item(), n=1)
                score_list.extend(outputs.cpu().detach().numpy())
                label_list.extend(labels.cpu().detach().numpy())
                name_list.extend(names)

                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss_q}'\
                    .format(batch=ii + 1, size=len(trainval_loaders[phase]), bt=batch_time.avg, eta=bar.eta_td, loss_q=loss_qa.avg)
                bar.next()
            bar.finish()

            score_list = np.array(score_list)
            label_list = np.array(label_list)

            plcc_w, srcc_w, krcc_w = score_correlation(score_list, label_list)
            plcc_s, srcc_s, krcc_s = temporal_pooling_correlation(name_list, score_list, label_list)

            if phase == 'train':
                scheduler.step()
                train_barometer['loss'].append(loss_qa.avg)
                train_barometer['PLCC_win'].append(plcc_w)
                train_barometer['SRCC_win'].append(srcc_w)
                train_barometer['KRCC_win'].append(krcc_w)
                train_barometer['PLCC_seq'].append(plcc_s)
                train_barometer['SRCC_seq'].append(srcc_s)
                train_barometer['KRCC_seq'].append(krcc_s)
            elif phase == 'valid':
                valid_barometer['loss'].append(loss_qa.avg)
                valid_barometer['PLCC_win'].append(plcc_w)
                valid_barometer['SRCC_win'].append(srcc_w)
                valid_barometer['KRCC_win'].append(krcc_w)
                valid_barometer['PLCC_seq'].append(plcc_s)
                valid_barometer['SRCC_seq'].append(srcc_s)
                valid_barometer['KRCC_seq'].append(krcc_s)
                visualize_pooling_prediction(name_list, score_list, label_list, os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'), i)
                # visualize_correlation_temporally(name_list, score_list, label_list, os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'), i)

            batch_time.reset()
            loss_qa.reset()

    # with open(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}', 'barometer.pkl'), mode='wb') as f:
    #     pickle.dump((train_barometer, valid_barometer), f)
    #     f.close()

    visualize_barometer(train_barometer, valid_barometer, os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'))

    return valid_barometer

if __name__ == "__main__":
    exp_name = '{year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}'.format(
        year=time.localtime().tm_year,
        month=time.localtime().tm_mon,
        day=time.localtime().tm_mday,
        hour=time.localtime().tm_hour,
        minute=time.localtime().tm_min,
        second=time.localtime().tm_sec
    )

    PLCC_win_list = np.zeros((NUM_REPETITIONS, num_epochs))
    SRCC_win_list = np.zeros((NUM_REPETITIONS, num_epochs))
    KRCC_win_list = np.zeros((NUM_REPETITIONS, num_epochs))
    PLCC_seq_list = np.zeros((NUM_REPETITIONS, num_epochs))
    SRCC_seq_list = np.zeros((NUM_REPETITIONS, num_epochs))
    KRCC_seq_list = np.zeros((NUM_REPETITIONS, num_epochs))
    for rep in range(NUM_REPETITIONS):
        barometer = train_model(exp_name, rep + 1)
        PLCC_win_list[rep, :] = barometer['PLCC_win']
        SRCC_win_list[rep, :] = barometer['SRCC_win']
        KRCC_win_list[rep, :] = barometer['KRCC_win']
        PLCC_seq_list[rep, :] = barometer['PLCC_seq']
        SRCC_seq_list[rep, :] = barometer['SRCC_seq']
        KRCC_seq_list[rep, :] = barometer['KRCC_seq']
        print('\n(Win-level) Average of {reps}/{total_reps} reps : PLCC / SRCC / KRCC = {plcc_m:.4f}±{plcc_s:.4f} {srcc_m:.4f}±{srcc_s:.4f} {krcc_m:.4f}±{krcc_s:.4f}'
            .format(reps=rep + 1,
                    total_reps=NUM_REPETITIONS,
                    plcc_m=PLCC_win_list[:rep + 1, -1].mean(),
                    srcc_m=SRCC_win_list[:rep + 1, -1].mean(),
                    krcc_m=KRCC_win_list[:rep + 1, -1].mean(),
                    plcc_s=PLCC_win_list[:rep + 1, -1].std(),
                    srcc_s=SRCC_win_list[:rep + 1, -1].std(),
                    krcc_s=KRCC_win_list[:rep + 1, -1].std()))
        print('(Vid-level) Average of {reps}/{total_reps} reps : PLCC / SRCC / KRCC = {plcc_m:.4f}±{plcc_s:.4f} {srcc_m:.4f}±{srcc_s:.4f} {krcc_m:.4f}±{krcc_s:.4f}\n'
            .format(reps=rep + 1,
                    total_reps=NUM_REPETITIONS,
                    plcc_m=PLCC_seq_list[:rep + 1, -1].mean(),
                    srcc_m=SRCC_seq_list[:rep + 1, -1].mean(),
                    krcc_m=KRCC_seq_list[:rep + 1, -1].mean(),
                    plcc_s=PLCC_seq_list[:rep + 1, -1].std(),
                    srcc_s=SRCC_seq_list[:rep + 1, -1].std(),
                    krcc_s=KRCC_seq_list[:rep + 1, -1].std()))
    np.save(os.path.join(EXP_FOLDER, exp_name, 'correlations.npy'), barometer)
