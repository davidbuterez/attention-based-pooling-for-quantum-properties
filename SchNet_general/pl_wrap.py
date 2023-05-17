import torch
import numpy as np
import pytorch_lightning as pl

from collections import defaultdict
from torch.nn import functional as F

# Imports from this project
from reporting import get_metrics
from schnet import SchNet


class Estimator(pl.LightningModule):
    def __init__(
            self,
            batch_size: int=32,
            lr: float=0.001,
            readout: str='linear',
            max_num_atoms_in_mol: int=55,
            scaler=None,
            monitor_loss: str='val_total_loss',
            name: str=None,
            aux_scaler=None,
            use_layer_norm=False,
            schnet_hidden_channels=128,
            schnet_num_filters=128,
            schnet_num_interactions=6,
            atomref=None,
            is_dipole=None,
            set_transformer_hidden_dim=None,
            set_transformer_num_heads=None,
            set_transformer_num_sabs=None,
        ):

        super().__init__()
        self.readout = readout
        self.lr = lr
        self.batch_size = batch_size
        self.max_num_atoms_in_mol = max_num_atoms_in_mol
        self.scaler = scaler
        self.aux_scaler = aux_scaler
        self.linear_output_size = 1
        self.monitor_loss = monitor_loss
        self.metric_fn = get_metrics
        self.name = name
        self.use_layer_norm = use_layer_norm

        self.schnet_hidden_channels = schnet_hidden_channels
        self.schnet_num_filters = schnet_num_filters
        self.schnet_num_interactions = schnet_num_interactions

        self.set_transformer_hidden_dim = set_transformer_hidden_dim
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_num_sabs = set_transformer_num_sabs

        self.atomref = atomref
        self.is_dipole = is_dipole

        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.test_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        self.net = SchNet(
            hidden_channels=self.schnet_hidden_channels,
            num_filters=self.schnet_num_filters,
            num_interactions=self.schnet_num_interactions,
            num_gaussians=50,
            cutoff=10.0,
            dipole=self.is_dipole,
            atomref=self.atomref,
            readout=self.readout,
            set_transformer_hidden_dim=self.set_transformer_hidden_dim,
            set_transformer_num_heads=self.set_transformer_num_heads,
            set_transformer_num_sabs=self.set_transformer_num_sabs,
            max_num_atoms_in_mol=self.max_num_atoms_in_mol
        )

    def forward(self, pos, atom_z, batch_mapping):
        predictions = self.net(pos=pos, z=atom_z, batch=batch_mapping)

        return predictions


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': opt,
            'monitor': self.monitor_loss
        }


    def _batch_loss(self, pos, y, atom_z, batch_mapping):
        predictions = self.forward(pos, atom_z, batch_mapping)

        loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y).float())

        return loss, predictions


    def _step(self, batch, step_type: str):
        assert step_type in ['train', 'valid', 'test']

        pos, y, atom_z, batch_mapping = batch.pos, batch.y, batch.z, batch.batch

        total_loss, predictions = self._batch_loss(pos, y, atom_z, batch_mapping)

        output = (torch.flatten(predictions), torch.flatten(y))

        if step_type == 'train':
            self.train_output[self.current_epoch].append(output)

        elif step_type == 'valid':
            self.val_output[self.current_epoch].append(output)

        elif step_type == 'test':
            self.test_output[self.num_called_test].append(output)

        return total_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_total_loss = self._step(batch, 'train')

        self.log('train_total_loss', train_total_loss, batch_size=self.batch_size)

        return train_total_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        val_total_loss = self._step(batch, 'valid')
        self.log('val_total_loss', val_total_loss, batch_size=self.batch_size)

        return val_total_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_total_loss = self._step(batch, 'test')

        self.log('test_total_loss', test_total_loss, batch_size=self.batch_size)

        return test_total_loss


    def _epoch_end_report(self, epoch_outputs, epoch_type):
        def flatten_list_of_tensors(lst):
            return np.array([item.item() for sublist in lst for item in sublist])

        y_pred = flatten_list_of_tensors([item[0] for item in epoch_outputs])
        y_true = flatten_list_of_tensors([item[1] for item in epoch_outputs])
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)

        if self.scaler:
            y_pred = self.scaler.inverse_transform(y_pred).squeeze()
            y_true = self.scaler.inverse_transform(y_true).squeeze()

        metrics = self.metric_fn(y_true, y_pred)
        self.log(f'{epoch_type} MAE', metrics[0], batch_size=self.batch_size)
        self.log(f'{epoch_type} RMSE', metrics[1], batch_size=self.batch_size)
        self.log(f'{epoch_type} R2', metrics[-1], batch_size=self.batch_size)

        return metrics, y_pred, y_true


    ### Do not save any training outputs
    def on_train_epoch_end(self):
        train_metrics, y_pred, y_true = self._epoch_end_report(self.train_output[self.current_epoch],
                                                               epoch_type='Train')

        self.train_metrics[self.current_epoch] = train_metrics

        del self.train_output[self.current_epoch]


    ### Do not save any validation outputs
    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            val_metrics, y_pred, y_true = self._epoch_end_report(self.val_output[self.current_epoch],
                                                                 epoch_type='Validation')

            self.val_metrics[self.current_epoch] = val_metrics

            del self.val_output[self.current_epoch]
        

    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        metrics, y_pred, y_true = self._epoch_end_report(test_outputs_per_epoch,
                                                         epoch_type='Test')

        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true
        self.test_metrics[self.num_called_test] = metrics

        self.num_called_test += 1
