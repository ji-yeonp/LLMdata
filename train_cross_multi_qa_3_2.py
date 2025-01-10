import time
import pickle
from utils.utils import *
from progress.bar import Bar
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.multimodal_dataset import MultimodalDataset
# from network.multi_feature_mae_token import MultiFeatureTransformer
# from network.multimodal_token import MultiModalTransformer
from network.cross_multi_st_former_3_2 import MultiModalTransformer

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_REPETITIONS = 50
num_epochs = 50
window_len = 256
num_tokens = 64
num_joints = 16
motion_channels = 6
music_channels = 35
input_embed = 'token'

embed_dim = 8
depth = 4
num_heads = 8
mlp_drop = 0.1

EXP_FOLDER = 'experiments_qa'
PRETRAIN_FOLDER = 'experiments_mae'
time_stamp = '2022-10-25-17-18-57'
exp_epoch = 'params_epoch-100.pth.tar'
pretrained = None # os.path.join(PRETRAIN_FOLDER, time_stamp, 'model_parameters', exp_epoch)

batch_size = 64
if pretrained is None:
    # train-from-scratch
    initial_lr = 1e-4
    w_decay = 3e-1
else:
    # fine-tune
    initial_lr = 1e-3
    w_decay = 5e-2
lr_decay = 0.99
resume_epoch = 0

def train_model(exp_name, repNo):

    if not os.path.exists(EXP_FOLDER):
        os.mkdir(EXP_FOLDER)
    if not os.path.exists(os.path.join(EXP_FOLDER, exp_name)):
        os.mkdir(os.path.join(EXP_FOLDER, exp_name))
    os.mkdir(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'))
    fout = open(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}', 'log.txt'), mode='w')

    model = MultiModalTransformer(
        window_len=window_len,
        num_tokens=num_tokens,
        in_channels=motion_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_drop=mlp_drop,
        input_embed=input_embed,
    )
    # model = MultiFeatureTransformer(
    #     window_len=window_len,
    #     num_tokens=num_tokens,
    #     in_channels=motion_channels,
    #     embed_dim=embed_dim,
    #     depth=depth,
    #     num_heads=num_heads,
    #     mlp_drop=mlp_drop,
    # )
    print(model)
    print(model, file=fout)

    if pretrained is None:
        print('Training from scratch...')
    else:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(pretrained))
        model.encoder.load_state_dict(checkpoint['encoder_dict'])

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=w_decay)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), weight_decay=w_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_decay ** epoch)

    print(optimizer)
    print(optimizer, file=fout)

    print('Total params: %.8fM' % (sum(p.numel() for p in model.parameters()) / 1.0e+6))
    print('Total params: %.8fM' % (sum(p.numel() for p in model.parameters()) / 1.0e+6), file=fout)
    model.to(device)
    criterion.to(device)

    train_dataloader = DataLoader(
        MultimodalDataset(dataset_path='data/Homogeneous_1', split='train', window_len=window_len, num_tokens=num_tokens),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
    )
    valid_dataloader = DataLoader(
        MultimodalDataset(dataset_path='data/Homogeneous_1', split='valid', window_len=window_len, num_tokens=num_tokens),
        batch_size=batch_size, pin_memory=True, num_workers=4,
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
        print('\nEpoch: {epoch}/{num_epochs} LR: {lr}'.format(epoch=i + 1, num_epochs=num_epochs, lr=curent_lr))
        print('\nEpoch: {epoch}/{num_epochs} LR: {lr}'.format(epoch=i + 1, num_epochs=num_epochs, lr=curent_lr), file=fout)

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
            for ii, (joints, bones, motions, accels, music, labels, names) in enumerate(trainval_loaders[phase]):
                # inputs_1 = joints.repeat(1, 1, 1, 2).to(device).float()
                # inputs_2 = bones.to(device).float()
                inputs_1 = bones.to(device).float()
                inputs_2 = motions.to(device).float()
                # inputs_1 = motions.to(device).float()
                # inputs_2 = joints.repeat(1, 1, 1, 2).to(device).float()
                music = music.to(device).float()
                labels = labels.to(device).float()

                if phase == 'train':
                    outputs = model(inputs_1, inputs_2, music)
                else:
                    with torch.no_grad():
                        outputs = model(inputs_1, inputs_2, music)

                loss = criterion(outputs, labels)
                loss_qa.update(loss.item(), n=1)

                score_list.extend(outputs.cpu().detach().numpy())
                label_list.extend(labels.cpu().detach().numpy())
                name_list.extend(names)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | ETA: {eta:} | Loss_q: {loss_q}'\
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

            # print('Vid-level : PLCC = {plcc:.4f} | SRCC = {srcc:.4f} | KRCC = {krcc:.4f}'.format(
            #     plcc=plcc, srcc=srcc, krcc=krcc))
            # print('{phase}: Loss = {loss:.6f} | PLCC = {plcc:.4f} | SRCC = {srcc:.4f} | KRCC = {krcc:.4f}'.format(
            #     phase=phase, loss=loss_qa.avg, plcc=plcc, srcc=srcc, krcc=krcc), file=fout)

            batch_time.reset()
            loss_qa.reset()

    with open(os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}', 'barometer.pkl'), mode='wb') as f:
        pickle.dump((train_barometer, valid_barometer), f)
        f.close()

    visualize_barometer(train_barometer, valid_barometer, os.path.join(EXP_FOLDER, exp_name, f'{repNo:03d}'))

    print('The log file is save at {file_name}'.format(file_name=exp_name))
    fout.close()

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
        print(
            '\n(Win-level) Average of {reps}/{total_reps} reps : PLCC / SRCC / KRCC = {plcc_m:.4f}±{plcc_s:.4f}   {srcc_m:.4f}±{srcc_s:.4f}   {krcc_m:.4f}±{krcc_s:.4f}'
            .format(reps=rep + 1,
                    total_reps=NUM_REPETITIONS,
                    plcc_m=PLCC_win_list[:rep + 1, -1].mean(),
                    srcc_m=SRCC_win_list[:rep + 1, -1].mean(),
                    krcc_m=KRCC_win_list[:rep + 1, -1].mean(),
                    plcc_s=PLCC_win_list[:rep + 1, -1].std(),
                    srcc_s=SRCC_win_list[:rep + 1, -1].std(),
                    krcc_s=KRCC_win_list[:rep + 1, -1].std()))
        print(
            '(Vid-level) Average of {reps}/{total_reps} reps : PLCC / SRCC / KRCC = {plcc_m:.4f}±{plcc_s:.4f}   {srcc_m:.4f}±{srcc_s:.4f}   {krcc_m:.4f}±{krcc_s:.4f}\n'
            .format(reps=rep + 1,
                    total_reps=NUM_REPETITIONS,
                    plcc_m=PLCC_seq_list[:rep + 1, -1].mean(),
                    srcc_m=SRCC_seq_list[:rep + 1, -1].mean(),
                    krcc_m=KRCC_seq_list[:rep + 1, -1].mean(),
                    plcc_s=PLCC_seq_list[:rep + 1, -1].std(),
                    srcc_s=SRCC_seq_list[:rep + 1, -1].std(),
                    krcc_s=KRCC_seq_list[:rep + 1, -1].std()))
    np.save(os.path.join(EXP_FOLDER, exp_name, 'correlations.npy'), barometer)