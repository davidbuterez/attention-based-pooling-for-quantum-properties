import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pathlib import Path

# Imports from this project
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from data_loading.load_QM7 import load_QM7
from data_loading.load_QM8 import load_QM8
from data_loading.load_QM9 import load_QM9
from data_loading.load_QMugs import load_QMugs
from data_loading.load_MD17 import load_MD17

from pl_wrap import Estimator


MAX_NUM_ATOMS_IN_MOL = {
    'QM7': 23,
    'QM8': 26,
    'QM9': 29,
    'QMugs': 228,
    'benzene': 12,
    'aspirin': 21,
    'malonaldehyde': 9,
    'ethanol': 9,
    'toluene': 15
}


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset-download-dir', type=str, required=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--readout', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--node-latent-dim', type=int, required=True)
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr', type=float, default=0.0001, required=False)
    parser.add_argument('--random-seed', type=int, required=True)
    parser.add_argument('--target-id', type=int, required=True)

    parser.add_argument('--set-transformer-hidden-dim', type=int, default=512, required=False)
    parser.add_argument('--set-transformer-num-heads', type=int, default=16, required=False)
    parser.add_argument('--set-transformer-num-sabs', type=int, default=2, required=False)

    parser.add_argument('--dimenet-hidden-channels', type=int, default=128)
    parser.add_argument('--dimenet-num-blocks', type=int, default=4)
    parser.add_argument('--dimenet-int-emb-size', type=int, default=64)
    parser.add_argument('--dimenet-basis-emb-size', type=int, default=8)
    parser.add_argument('--dimenet-out-emb-channels', type=int, default=256)

    parser.add_argument('--ckpt-path', type=str, required=False, default=None)

    args = parser.parse_args()
    argsdict = vars(args)

    SEED = 0
    learning_rate = argsdict['lr']
    batch_size = argsdict['batch_size']

    dimenet_hidden_channels = argsdict['dimenet_hidden_channels']
    dimenet_num_blocks = argsdict['dimenet_num_blocks']
    dimenet_int_emb_size = argsdict['dimenet_int_emb_size']
    dimenet_basis_emb_size = argsdict['dimenet_basis_emb_size']
    dimenet_out_emb_channels = argsdict['dimenet_out_emb_channels']

    set_transformer_hidden_dim = argsdict['set_transformer_hidden_dim']
    set_transformer_num_heads = argsdict['set_transformer_num_heads']
    set_transformer_num_sabs = argsdict['set_transformer_num_sabs']

    random_seed = argsdict['random_seed']
    target_id = argsdict['target_id']
    readout = argsdict['readout']

    pl.seed_everything(SEED)

    dataset = argsdict['dataset']
    assert dataset in ['QM7', 'QM8', 'QM9', 'QMugs', 'benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']
    print('Loading dataset...')

    NUM_WORKERS = 12

    if dataset == 'QM7':
        train, val, test, scaler = load_QM7(argsdict['random_seed'], argsdict['target_id'])

    elif dataset == 'QM8':
        train, val, test, scaler = load_QM8(argsdict['random_seed'], argsdict['target_id'])

    elif dataset == 'QM9':
        train, val, test, scaler = load_QM9(argsdict['random_seed'], argsdict['target_id'],
                                    download_dir=argsdict['dataset_download_dir'])

    elif dataset == 'QMugs':
        train, val, test, scaler = load_QMugs(argsdict['random_seed'], argsdict['target_id'])

    elif dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        train, val, test, scaler = load_MD17(ds=dataset, download_dir=argsdict['dataset_download_dir'])
        NUM_WORKERS = 0

    train_dataloader = GeometricDataLoader(train, batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = GeometricDataLoader(val, batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = GeometricDataLoader(test, batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print('Loaded dataset!')


    NAME = f'DATASET={dataset}+target_id={target_id}+random_seed={random_seed}+readout={readout}+lr={learning_rate}'
    NAME += f'+dn_hd_cnl={dimenet_hidden_channels}+dn_num_blk={dimenet_num_blocks}+dn_int_emb_sz={dimenet_int_emb_size}'
    NAME += f'+dn_basis_emb_sz={dimenet_basis_emb_size}+dn_out_emb_cnl={dimenet_out_emb_channels}'

    if readout == 'set_transformer':
        NAME += f'+set_transformer_num_SABs={set_transformer_num_sabs}+set_transformer_hidden_dim={set_transformer_hidden_dim}'
        NAME += f'+set_transformer_num_heads={set_transformer_num_heads}'

    OUT_DIR = os.path.join(argsdict['out_dir'], f'DimeNet++/{dataset}/{random_seed}/{target_id}/{readout}', NAME)
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    gnn_args = dict(
        readout=argsdict['readout'],
        batch_size=batch_size,
        lr=learning_rate,
        max_num_atoms_in_mol=MAX_NUM_ATOMS_IN_MOL[dataset],
        scaler=scaler,
        use_layer_norm=False,
        node_latent_dim=argsdict['node_latent_dim'],
        dimenet_hidden_channels=dimenet_hidden_channels,
        dimenet_num_blocks=dimenet_num_blocks,
        dimenet_int_emb_size=dimenet_int_emb_size,
        dimenet_basis_emb_size=dimenet_basis_emb_size,
        dimenet_out_emb_channels=dimenet_out_emb_channels,
        set_transformer_hidden_dim=set_transformer_hidden_dim,
        set_transformer_num_heads=set_transformer_num_heads,
        set_transformer_num_sabs=set_transformer_num_sabs
    )


    model = Estimator(**gnn_args)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_loss',
        dirpath=OUT_DIR,
        filename='{epoch:04d}',
        mode='min',
        save_top_k=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_total_loss',
        patience=10,
        mode='min'
    )

    # Logger
    logger = WandbLogger(
        name=NAME,
        project='Modelling local and general quantum mechanical properties with attention-based pooling'
    )

    # Trainer
    common_trainer_args = dict(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        num_sanity_val_steps=0,
        devices=1,
        min_epochs=1,
        max_epochs=-1
    )

    if argsdict['use_cuda']:
        model = model.cuda()
        trainer_args = common_trainer_args | dict(accelerator='gpu')
    else:
        trainer_args = common_trainer_args | dict(accelerator='cpu')


    if argsdict['ckpt_path']:
        trainer_args = dict(resume_from_checkpoint=argsdict['ckpt_path']) | trainer_args

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

    # Save test metrics
    np.save(os.path.join(OUT_DIR, 'test_y_pred.npy'), model.test_output)
    np.save(os.path.join(OUT_DIR, 'test_y_true.npy'), model.test_true)
    np.save(os.path.join(OUT_DIR, 'test_metrics.npy'), model.test_metrics)


if __name__ == "__main__":
    main()
