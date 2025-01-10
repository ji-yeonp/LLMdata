import pickle
import time
from utils.utils import *
from progress.bar import Bar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# from dataloaders.optitrack_dataset_sample_0 import OptiTrackDataset
# from dataloaders.optitrack_dataset_sample_1_1 import OptiTrackDataset
from dataloaders.optitrack_dataset_sample_4_1_0 import OptiTrackDataset
from network_mae.temporal_former_0_0 import *


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = 1

DENOMINATOR = 10
STRIDE = 6
FPS = 360

NUM_REPETITIONS = 10
num_epochs = 10
window_frames = 256
num_tokens = 64
num_joints = 21
motion_channels = 3
music_channels = 35

embed_dim = 4
depth = 4
num_heads = 1
mlp_drop = 0.0

NUM_SPLITS = 2
EXP_FOLDER = 'experiments-qa'
PRETRAIN_FOLDER = 'experiments-mae'
# time_stamp = '2023-08-11-22-33-34' # Joint feature
time_stamp = '2023-08-31-17-38-37' # Velo feature
exp_epoch = 'best-model-params.pth.tar'
pretrained = os.path.join(PRETRAIN_FOLDER, time_stamp, exp_epoch)

batch_size = 1024 * NUM_GPUS
initial_lr = 1e-3
w_decay = 5e-2
lr_decay = 0.9
resume_epoch = 0

# Antifragile, Attention, PinkVenom, Shutdown
# TRAIN_SET = ['Attention', 'PinkVenom', 'Shutdown']
# VALID_SET = ['Antifragile']
TRAIN_SET = ['Antifragile', 'Attention']
UPDATE_SET = ['PinkVenom']
VALID_SET = ['Shutdown']

def train_model(exp_name, repNo):
    if not os.path.exists(EXP_FOLDER):
        os.mkdir(EXP_FOLDER)
    if not os.path.exists(os.path.join(EXP_FOLDER, exp_name)):
        os.mkdir(os.path.join(EXP_FOLDER, exp_name))
    if not os.path.exists(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}')):
        os.mkdir(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'))
        
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
    print(model)

    if pretrained is None:
        print('Training from scratch...')
    else:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(pretrained))
        model.encoder.load_state_dict(checkpoint['encoder_params'])

    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    
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
            denominator=DENOMINATOR,
            max_stride=STRIDE,
        ),
        batch_size=batch_size, shuffle=True, num_workers=4 * NUM_GPUS,
    )
    update_dataloader = DataLoader(
        OptiTrackDataset(
            dataset_path=[f'./data/{name}' for name in UPDATE_SET],
            split='update',
            window_frames=window_frames,
            denominator=DENOMINATOR,
            max_stride=STRIDE,
        ),
        batch_size=batch_size, num_workers=1 * NUM_GPUS,
    )
    valid_dataloader = DataLoader(
        OptiTrackDataset(
            dataset_path=[f'./data/{name}' for name in VALID_SET],
            split='valid',
            window_frames=window_frames,
            denominator=DENOMINATOR,
            max_stride=STRIDE,
        ),
        batch_size=batch_size, num_workers=1 * NUM_GPUS,
    )
    trainval_loaders = {
        'train': train_dataloader,
        'update': update_dataloader,
        'valid': valid_dataloader,
    }

    loss_qa = AverageMeter()
    batch_time = AverageMeter()
    train_barometer = {
        'loss': list(),
        'PLCC_win': list(),
        'SRCC_win': list(),
        'KRCC_win': list(),
        'CoD_win': list(),
        'PLCC_seq': list(),
        'SRCC_seq': list(),
        'KRCC_seq': list(),
        'CoD_seq': list(),
    }
    update_barometer = {
        'loss': list(),
        'PLCC_win': list(),
        'SRCC_win': list(),
        'KRCC_win': list(),
        'CoD_win': list(),
        'PLCC_seq': list(),
        'SRCC_seq': list(),
        'KRCC_seq': list(),
        'CoD_seq': list(),
    }
    valid_barometer = {
        'loss': list(),
        'PLCC_win': list(),
        'SRCC_win': list(),
        'KRCC_win': list(),
        'CoD_win': list(),
        'PLCC_seq': list(),
        'SRCC_seq': list(),
        'KRCC_seq': list(),
        'CoD_seq': list(),
    }

    for i in range(resume_epoch, num_epochs):
        curent_lr = scheduler.get_last_lr()[0]
        print('\nEpoch: {epoch}/{num_epochs} LR: {lr:.8f}'.format(epoch=i + 1, num_epochs=num_epochs, lr=curent_lr))

        for phase in trainval_loaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            end = time.time()
            bar = Bar(phase, max=len(trainval_loaders[phase]))
            score_list = list()
            label_list = list()
            name_list = list()
            for ii, (joints, bones, motions, labels, names, _) in enumerate(trainval_loaders[phase]):
                inputs = motions.to(device).float()
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

            plcc_w, srcc_w, krcc_w, cod_w = score_correlation(score_list, label_list)
            plcc_s, srcc_s, krcc_s, cod_s = temporal_pooling_correlation(name_list, score_list, label_list)

            if phase == 'train':
                scheduler.step()

                train_barometer['loss'].append(loss_qa.avg)
                train_barometer['PLCC_win'].append(plcc_w)
                train_barometer['SRCC_win'].append(srcc_w)
                train_barometer['KRCC_win'].append(krcc_w)
                train_barometer['CoD_win'].append(cod_w)

                train_barometer['PLCC_seq'].append(plcc_s)
                train_barometer['SRCC_seq'].append(srcc_s)
                train_barometer['KRCC_seq'].append(krcc_s)
                train_barometer['CoD_seq'].append(cod_s)
            elif phase == 'update':
                update_barometer['loss'].append(loss_qa.avg)

                update_barometer['PLCC_win'].append(plcc_w)
                update_barometer['SRCC_win'].append(srcc_w)
                update_barometer['KRCC_win'].append(krcc_w)
                update_barometer['CoD_win'].append(cod_w)

                update_barometer['PLCC_seq'].append(plcc_s)
                update_barometer['SRCC_seq'].append(srcc_s)
                update_barometer['KRCC_seq'].append(krcc_s)
                update_barometer['CoD_seq'].append(cod_s)

                visualize_pooling_prediction(name_list, score_list, label_list, os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'), i)
                # visualize_correlation_temporally(name_list, score_list, label_list, os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'), i)
                # diversity_measure(name_list, score_list, label_list, f'update-measure-sample001-str{STRIDE}.npy')
            elif phase == 'valid':
                valid_barometer['loss'].append(loss_qa.avg)

                valid_barometer['PLCC_win'].append(plcc_w)
                valid_barometer['SRCC_win'].append(srcc_w)
                valid_barometer['KRCC_win'].append(krcc_w)
                valid_barometer['CoD_win'].append(cod_w)

                valid_barometer['PLCC_seq'].append(plcc_s)
                valid_barometer['SRCC_seq'].append(srcc_s)
                valid_barometer['KRCC_seq'].append(krcc_s)
                valid_barometer['CoD_seq'].append(cod_s)

                visualize_pooling_prediction(name_list, score_list, label_list, os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'), i)
                # diversity_measure(name_list, score_list, label_list)

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
    CoD_win_list = np.zeros((NUM_REPETITIONS, num_epochs))

    PLCC_seq_list = np.zeros((NUM_REPETITIONS, num_epochs))
    SRCC_seq_list = np.zeros((NUM_REPETITIONS, num_epochs))
    KRCC_seq_list = np.zeros((NUM_REPETITIONS, num_epochs))
    CoD_seq_list = np.zeros((NUM_REPETITIONS, num_epochs))

    for rep in range(NUM_REPETITIONS):
        barometer = train_model(exp_name, rep + 1)

        PLCC_win_list[rep, :] = barometer['PLCC_win']
        SRCC_win_list[rep, :] = barometer['SRCC_win']
        KRCC_win_list[rep, :] = barometer['KRCC_win']
        CoD_win_list[rep, :] = barometer['CoD_win']

        PLCC_seq_list[rep, :] = barometer['PLCC_seq']
        SRCC_seq_list[rep, :] = barometer['SRCC_seq']
        KRCC_seq_list[rep, :] = barometer['KRCC_seq']
        CoD_seq_list[rep, :] = barometer['CoD_seq']

        print_str = f'\n(Win-level) Average of {rep + 1}/{NUM_REPETITIONS} reps : PLCC / SRCC / KRCC / CoD ='
        print_str += f' {PLCC_win_list[:rep + 1, -1].mean():.4f}±{PLCC_win_list[:rep + 1, -1].std():.4f}'
        print_str += f' {SRCC_win_list[:rep + 1, -1].mean():.4f}±{SRCC_win_list[:rep + 1, -1].std():.4f}'
        print_str += f' {KRCC_win_list[:rep + 1, -1].mean():.4f}±{KRCC_win_list[:rep + 1, -1].std():.4f}'
        print_str += f' {CoD_win_list[:rep + 1, -1].mean():.4f}±{CoD_win_list[:rep + 1, -1].std():.4f}'
        print(print_str)

        print_str = f'(Seq-level) Average of {rep + 1}/{NUM_REPETITIONS} reps : PLCC / SRCC / KRCC / CoD ='
        print_str += f' {PLCC_seq_list[:rep + 1, -1].mean():.4f}±{PLCC_seq_list[:rep + 1, -1].std():.4f}'
        print_str += f' {SRCC_seq_list[:rep + 1, -1].mean():.4f}±{SRCC_seq_list[:rep + 1, -1].std():.4f}'
        print_str += f' {KRCC_seq_list[:rep + 1, -1].mean():.4f}±{KRCC_seq_list[:rep + 1, -1].std():.4f}'
        print_str += f' {CoD_seq_list[:rep + 1, -1].mean():.4f}±{CoD_seq_list[:rep + 1, -1].std():.4f}'
        print(print_str)

        print_str = f'\n(Win-level) Average of {rep + 1}/{NUM_REPETITIONS} reps : PLCC / SRCC / KRCC / CoD ='
        print_str += f' {PLCC_win_list[:rep + 1, :].max():.4f}'
        print_str += f' {SRCC_win_list[:rep + 1, :].max():.4f}'
        print_str += f' {KRCC_win_list[:rep + 1, :].max():.4f}'
        print_str += f' {CoD_win_list[:rep + 1, :].max():.4f}'
        print(print_str)

        print_str = f'(Seq-level) Average of {rep + 1}/{NUM_REPETITIONS} reps : PLCC / SRCC / KRCC / CoD ='
        print_str += f' {PLCC_seq_list[:rep + 1, :].max():.4f}'
        print_str += f' {SRCC_seq_list[:rep + 1, :].max():.4f}'
        print_str += f' {KRCC_seq_list[:rep + 1, :].max():.4f}'
        print_str += f' {CoD_seq_list[:rep + 1, :].max():.4f}'
        print(print_str)

    np.save(os.path.join(EXP_FOLDER, exp_name, 'correlations.npy'), barometer)
