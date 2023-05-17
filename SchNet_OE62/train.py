import argparse
import os
import torch
import wandb
import schnetpack as spk
from pathlib import Path
from torch.optim import Adam
from schnetpack import AtomsData
from schnet_trainer import Trainer
from schnetpack.train import CSVHook, ReduceLROnPlateauHook, EarlyStoppingHook
from schnetpack.train.metrics import MeanAbsoluteError
from atomwise import Atomwise, Atomwise_WA


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str)
parser.add_argument('--random-split-idx', type=int)
parser.add_argument('--task', choices=['HOMO', 'LUMO'], type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--readout', choices=['sum', 'avg', 'max', 'owa', 'wa', 'set_transformer'], type=str)
parser.add_argument('--out-dir', type=str)
parser.add_argument('--load-from-stored-checkpoint', action='store_true')
parser.add_argument('--no-load-from-stored-checkpoint', dest='load_from_stored_ckpt', action='store_false')
parser.add_argument('--set-transformer-hidden-dim', type=int)
parser.add_argument('--set-transformer-num-heads', type=int)
parser.add_argument('--set-transformer-num-sabs', type=int)
parser.set_defaults(load_from_stored_ckpt=False)

args = parser.parse_args()
argsdict = vars(args)

st_hidden_dim = argsdict['set_transformer_hidden_dim']
st_num_heads = argsdict['set_transformer_num_heads']
st_num_sabs = argsdict['set_transformer_num_sabs']
random_split_idx = argsdict['random_split_idx']
batch_size = argsdict['batch_size']
task = argsdict['task']
readout = argsdict['readout']

LR = 0.0001

NAME = f'DATASET=OE62+random-split-idx={random_split_idx}+task={task}+readout={readout}+batch-size={batch_size}'
NAME += f'+LR={LR}'
if readout == 'set_transformer':
    NAME += f'+set_transformer_num_SABs={st_num_sabs}+set_transformer_hidden_dim={st_hidden_dim}'
    NAME += f'+set_transformer_num_heads={st_num_heads}'

out_dir = os.path.join(argsdict['out_dir'], NAME)
Path(out_dir).mkdir(exist_ok=True, parents=True)

run = wandb.init(
    project='Modelling local and general quantum mechanical properties with attention-based pooling - OE62',
    name=NAME
)

print(f'Training on OE62 split {random_split_idx}: task = {task}, readout = {readout}, batch size = {batch_size},', end='')
print(f' learning rate = {LR}')
if readout == 'set_transformer':
    print(f'Set Transformer configuration: hidden dimension = {st_hidden_dim},', end='')
    print(f' # heads = {st_num_heads}, # SABs = {st_num_sabs}')

dataset = AtomsData(
    argsdict['data_path'],
    available_properties=['HOMO','HOMO_au','LUMO','GAP','IP','IP_au','coeff']
)

train, val, test = spk.train_test_split(
   dataset, 32000, 19480, f'splits/split_32000_19480_10000_{random_split_idx}.npz'
)

train_loader = spk.AtomsLoader(train, batch_size=batch_size, num_workers=12, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=batch_size, num_workers=12)
test_loader = spk.AtomsLoader(test, batch_size=batch_size, num_workers=12)

def mse_loss(batch, result):
    err_sq = torch.nn.functional.mse_loss(batch[task], result[task])
    return err_sq

def mse_loss_owa(batch, result):
    err_sq = 0.1 * torch.nn.functional.mse_loss(batch[task], result[task]) + \
            torch.nn.functional.mse_loss(batch['coeff'], result['weight'].squeeze(-1))
    return err_sq

properties = ['HOMO','HOMO_au','LUMO','GAP','IP','IP_au','coeff']

means, stddevs = train_loader.get_statistics([task])

representation = spk.representation.SchNet(
    n_atom_basis=128,
    n_filters=128,
    n_gaussians=64,
    n_interactions=6,
    cutoff=5,
    cutoff_network=spk.nn.cutoff.CosineCutoff
)

if readout not in ['owa', 'wa']:
    property_out = Atomwise(
        n_in=128,
        n_out=1 if readout != 'set_transformer' else 128,
        aggregation_mode=readout,
        property=task,
        mean=means[task],
        stddev=stddevs[task],
        n_layers=4,
        set_transformer_hidden_dim=st_hidden_dim,
        set_transformer_num_heads=st_num_heads,
        set_transformer_num_sabs=st_num_sabs
    )
else:
    property_out = Atomwise_WA(
        n_in=128,
        n_out=1,
        property=task,
        mean=means[task],
        stddev=stddevs[task],
        n_layers=4,
    )

model = spk.atomistic.AtomisticModel(representation, property_out)

optimizer = Adam(params=model.parameters(), lr=LR)

metrics = [MeanAbsoluteError(p, p) for p in [task]]

hooks = [
    CSVHook(log_path=out_dir, metrics=metrics),
    ReduceLROnPlateauHook(optimizer, patience=10, factor=0.8, min_lr=1e-6, stop_after_min=False),
    EarlyStoppingHook(patience=60)
]

trainer = Trainer(
    out_dir,
    model=model,
    hooks=hooks,
    loss_fn=mse_loss_owa if readout == 'owa' else mse_loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    validation_test_loader=test_loader,
    validation_interval=1,
    property_name=task,
    load_from_stored_ckpt=argsdict['load_from_stored_checkpoint']
)

trainer.train(device='cuda', n_epochs=1500, model_dir=out_dir)
