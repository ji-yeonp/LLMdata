import time
import redis
from redis.commands.search.field import NumericField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition
from redis.commands.search.query import Query

from utils.utils import *
from progress.bar import Bar

import torch
from torch.utils.data import DataLoader

from dataloaders.database_dataset import DatabaseDataset
from network.unimodal_mae_token import MaskedAutoencoder


torch.multiprocessing.set_sharing_strategy('file_system')

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

window_len = 256
num_tokens = 64
num_joints = 16
motion_channels = 3

batch_size = 4096
embed_dim = 6
depth = 4
num_heads = 6
decoder_embed_dim = 3
decoder_depth = 2
decoder_num_heads = 3
mask_ratio = 0.75

PRETRAIN_FOLDER = 'experiments_mae'
time_stamp = '2022-11-30-19-37-03'
epochNo = 200
exp_epoch = f'params_epoch-{epochNo:03d}.pth.tar'
pretrained = os.path.join(PRETRAIN_FOLDER, time_stamp, 'model_parameters', exp_epoch)

def process_db():

    model = MaskedAutoencoder(
        window_len=window_len,
        num_tokens=num_tokens,
        in_channels=motion_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
    )
    print(model)
    print('Total params: %.8fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if pretrained is None:
        print('Training from scratch...')
    else:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        model.encoder.load_state_dict(checkpoint['encoder_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_dict'])
        print("Initializing weights from: {}...".format(pretrained))

    train_dataloader = DataLoader(
        DatabaseDataset(dataset_path='data/pretrain_2', preprocess='data_stats_joint', split='train', window_len=window_len, num_tokens=num_tokens),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    valid_dataloader = DataLoader(
        DatabaseDataset(dataset_path='data/pretrain_2', preprocess='data_stats_joint', split='valid', window_len=window_len, num_tokens=num_tokens),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    trainval_loaders = {'train': train_dataloader, 'valid': valid_dataloader}

    train_dict = {
        'name': list(),
        'input': list(),
        'feature': list(),
        'frame_f': list(),
        'frame_l': list(),
        'loss': list(),
    }

    valid_dict = {
        'name': list(),
        'input': list(),
        'feature': list(),
        'frame_f': list(),
        'frame_l': list(),
        'loss': list(),
    }

    batch_time = AverageMeter()
    loss_mae = AverageMeter()
    model.eval()
    end = time.time()
    for phase in ['train', 'valid']:
        bar = Bar(phase, max=len(trainval_loaders[phase]))
        for ii, (motions, _, names, frms_0, frms_1) in enumerate(trainval_loaders[phase]):
            motions = motions.to(device)

            with torch.no_grad():
                # if mask_ratio == 0.0:
                #     y = model.encoder(motions, mask_ratio=mask_ratio)
                # else:
                #     y, _, _= model.encoder(motions, mask_ratio=mask_ratio)
                loss, h, _ = model(motions, mask_ratio=mask_ratio)

            if phase == 'train':
                train_dict['name'].extend(names)
                # train_dict['input'].append(motions.cpu().detach().numpy())
                train_dict['feature'].append(h[:, 1:, :].mean(dim=1).cpu().detach().numpy())
                train_dict['frame_f'].append(frms_0.cpu().detach().numpy())
                train_dict['frame_l'].append(frms_1.cpu().detach().numpy())
                # train_dict['loss'].append(loss.cpu().detach().numpy())
            elif phase == 'valid':
                valid_dict['name'].extend(names)
                # valid_dict['input'].append(motions.cpu().detach().numpy())
                valid_dict['feature'].append(h[:, 1:, :].mean(dim=1).cpu().detach().numpy())
                valid_dict['frame_f'].append(frms_0.cpu().detach().numpy())
                valid_dict['frame_l'].append(frms_1.cpu().detach().numpy())
                # valid_dict['loss'].append(loss.cpu().detach().numpy())
            else:
                pass

            loss_mae.update(loss.mean().item(), n=1)
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss}'\
                .format(batch=ii + 1, size=len(trainval_loaders[phase]), bt=batch_time.avg, eta=bar.eta_td, loss=loss_mae.avg)
            bar.next()
        bar.finish()

        batch_time.reset()
        loss_mae.reset()

    train_dict['feature'] = np.concatenate(train_dict['feature'])
    valid_dict['feature'] = np.concatenate(valid_dict['feature'])

    train_dict['frame_f'] = np.concatenate(train_dict['frame_f'])
    valid_dict['frame_f'] = np.concatenate(valid_dict['frame_f'])

    print(len(train_dict['name']), train_dict['feature'].shape, train_dict['frame_f'].shape)
    print(len(valid_dict['name']), valid_dict['feature'].shape, valid_dict['frame_f'].shape)

    # with open('database/train_features.csv', mode='w', newline='') as f:
    #     writer = csv.writer(f)
    #     for ff in train_dict['feature'][::10, :]:
    #         writer.writerow(ff)
    #     f.close()
    #
    # with open('database/valid_features.csv', mode='w', newline='') as f:
    #     writer = csv.writer(f)
    #     for ff in valid_dict['feature'][::10, :]:
    #         writer.writerow(ff)
    #     f.close()

    ####################################################################################################################
    # # Motion Identification
    # train_features = torch.from_numpy(train_dict['feature']).to(device)
    # valid_features = torch.from_numpy(valid_dict['feature']).to(device)
    #
    # results = list()
    # batch_time = AverageMeter()
    # accuracy = AverageMeter()
    # bar = Bar(f'identify', max=len(valid_dict['name']))
    # for i, (n, fi, f) in enumerate(zip(valid_dict['name'], valid_dict['frame_f'], valid_features)):
    #     end = time.time()
    #
    #     query = f.tile((train_features.size(0), 1))
    #     distance = torch.linalg.norm(query - train_features, ord=2, dim=1)
    #     dist_rank = torch.argsort(distance)
    #     # print(query.size(), distance.size(), dist_rank.size(), len(train_dict['name']))
    #
    #     batch_time.update(time.time() - end)
    #     accuracy.update(100.0 if n == train_dict['name'][dist_rank[0]] else 0.0)
    #
    #     results.append({
    #         'query_name': n,
    #         'query_frame': fi,
    #         'match_distance': distance[i].item(),
    #         'result_name': train_dict['name'][dist_rank[0]],
    #         'result_frame': train_dict['frame_f'][dist_rank[0]],
    #         'result_distance': distance[dist_rank[0]].item(),
    #     })
    #
    #     bar.suffix = '({batch}/{size}) | Batch: {bt:.8f}s | ETA: {eta:} | Acc: {acc}' \
    #         .format(batch=i + 1, size=len(valid_dict['name']), bt=batch_time.avg, eta=bar.eta_td, acc=accuracy.avg)
    #     bar.next()
    # bar.finish()

    # # with open(f'results/{folder}_e{epochNo:03d}_{mask_ratio:03d}.pkl', mode='wb') as f:
    # #     pickle.dump((results), f)
    ####################################################################################################################

    # train_frame_info = np.concatenate(train_frame_info)
    # valid_frame_info = np.concatenate(valid_frame_info)
    #
    # with open(f'database/train_data_t{num_tokens}_{epochNo:03d}_{int(100 * mask_ratio):03d}.pkl', mode='wb') as f:
    #     pickle.dump((train_names, train_frame_info, train_features), f)
    #
    # with open(f'database/valid_data_t{num_tokens}_{epochNo:03d}_{int(100 * mask_ratio):03d}.pkl', mode='wb') as f:
    #     pickle.dump((valid_names, valid_frame_info, valid_features), f)
    #
    # print(train_features.shape, len(train_names), train_frame_info.shape)
    # print(valid_features.shape, len(valid_names), valid_frame_info.shape)
    #
    # centroids = np.load(os.path.join(PRETRAIN_FOLDER, time_stamp, 'mae_recon', f'c-epoch-{epochNo:03d}.npy'))
    # print(centroids.shape)
    #
    # cluster_index = list()
    # cluster_check = dict()
    # for feat in train_features:
    #     new_feat = np.tile(feat, (centroids.shape[0], 1))
    #     distance = np.linalg.norm(new_feat - centroids, ord=2, axis=1)
    #     dist_rank = np.argmin(distance)
    #     cluster_index.append(f'{dist_rank:02d}')
    #
    #     if f'{dist_rank:02d}' in cluster_check.keys():
    #         pass
    #     else:
    #         cluster_check[f'{dist_rank:02d}'] = True
    # print(cluster_check)
    #
    # end = time.time()
    # plot_vecs_n_labels(train_features, cluster_index, len(cluster_check.keys()), centroids, os.path.join('database', f't-sne-e{epochNo:03d}.png'))
    # print(f'{(time.time() - end):.2f} sec')

    with redis.StrictRedis(host='localhost', port=6379, db=0) as rd:
        rd.ping()
        rd.flushdb()
        schema = (
            VectorField(
                'fingerprint',
                'FLAT',
                {'TYPE': 'FLOAT32', 'DIM': DIM_DATA, 'DISTANCE_METRIC': 'L2'}
            ),
        )
        res = rd.ft('random_idx').create_index(schema)

        proc_time = AverageMeter()
        bar = Bar('insert', max=len(train_dict['name']))
        for ii, (name, info, feat) in enumerate(zip(train_dict['name'], train_dict['frame_f'], train_dict['feature'])):
            end = time.time()
            #####################################################################################
            data_dict = {
                'frame_0': f'{info:04d}',
                'window_length': window_len,
                'feature': feat.astype(np.float32).tobytes(),
            }
            rd.hset(f'{name}_{info:04d}', mapping=data_dict)
            #####################################################################################
            proc_time.update(time.time() - end)
            bar.suffix = '({batch}/{size}) | Batch: {bt}s'.format(batch=ii + 1, size=len(train_dict['name']), bt=proc_time.avg)
            bar.next()
        bar.finish()

if __name__ == "__main__":
    process_db()
