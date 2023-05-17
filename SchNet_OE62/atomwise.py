import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import grad

import schnetpack
from schnetpack import nn as L, Properties


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            # self.ln0 = nn.LayerNorm(dim_V)
            # self.ln1 = nn.LayerNorm(dim_V)
            self.ln0 = nn.BatchNorm1d(dim_V)
            self.ln1 = nn.BatchNorm1d(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V), 2)
        # A = F.dropout(A, p=0.1)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O.permute(0, 2, 1)).permute(0, 2, 1)
        O = O + F.relu(self.fc_o(O))
        # O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O.permute(0, 2, 1)).permute(0, 2, 1)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False, num_sabs=2):
        super(SetTransformer, self).__init__()
        
        if num_sabs == 2:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                # nn.Dropout(p=0.1),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                # nn.Dropout(p=0.1),
            )
        elif num_sabs == 3:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                # nn.Dropout(p=0.1),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                # nn.Dropout(p=0.1),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                # nn.Dropout(p=0.1),
            )
            
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                # nn.Dropout(p=0.1),
                nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        return self.dec(self.enc(X))


class AtomwiseError(Exception):
    pass


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        stress (str or None): Name of stress property. Compute the derivative with
            respect to the cell parameters if not None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not
            needed and often can be worked around in a much more efficient way.
            Defaults to the value of create_graph. (default: False)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=False,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
        set_transformer_hidden_dim: int=256,
        set_transformer_num_heads: int=8,
        set_transformer_num_sabs: int=2,
        max_num_atoms_in_mol: int=1
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.stress = stress

        self.aggregation_mode = aggregation_mode
        self.set_transformer_hidden_dim = set_transformer_hidden_dim
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_num_sabs = set_transformer_num_sabs
        self.max_num_atoms_in_mol = max_num_atoms_in_mol

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref.astype(np.float32))
            )
        else:
            self.atomref = None

        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
        elif aggregation_mode == "max":
            self.atom_pool = schnetpack.nn.base.MaxAggregate(axis=1)
        elif aggregation_mode == "softmax":
            self.atom_pool = schnetpack.nn.base.SoftmaxAggregate(axis=1)
        elif aggregation_mode == 'set_transformer':
            self.atom_pool = SetTransformer(dim_input=n_out, num_outputs=32,
                            dim_output=n_out, num_inds=None, dim_hidden=self.set_transformer_hidden_dim,
                            num_heads=self.set_transformer_num_heads, ln=False, num_sabs=self.set_transformer_num_sabs)
            self.regr_nn = torch.nn.Sequential(
                    nn.Linear(n_out, n_out // 2),
                    ShiftedSoftplus(),
                    nn.Linear(n_out // 2, n_out // 2),
                    ShiftedSoftplus(),
                    nn.Linear(n_out // 2, 1)
                )
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0


        if self.aggregation_mode != 'set_transformer':
            y = self.atom_pool(yi, atom_mask)
        else:
            y_masked = yi * atom_mask[..., None]
            y = self.atom_pool(y_masked)
            y = y.mean(dim=1)
            y = self.regr_nn(y)

        # collect results
        result = {self.property: y}

        if self.contributions is not None:
            result[self.contributions] = yi

        create_graph = True if self.training else self.create_graph

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy

        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume

        return result


class Atomwise_WA(nn.Module):
    def __init__(
        self,
        n_in,
        n_out=2,
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        property='HOMO_au',
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=False,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.stress = stress

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev
        #for homo prediciton
        self.outnet = nn.Sequential(
            schnetpack.nn.base.GetItem("representation"),
            schnetpack.nn.blocks.MLP(n_in,n_out,n_neurons,n_layers, activation),
        )
        #for atomic coeff prediciton
        self.outnet1 = nn.Sequential(
            schnetpack.nn.base.GetItem("representation"),
            schnetpack.nn.blocks.MLP(n_in,1,n_neurons,n_layers, activation),
        )

        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)
        #self.standardize_k = schnetpack.nn.base.ScaleShift(torch.FloatTensor([1.75]),torch.FloatTensor([0.2]))
        self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)


    def forward(self,inputs):
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]
        y1 = self.outnet1(inputs)
        weight = softmax(y1, atom_mask)
        y2 = self.outnet(inputs)
        y = self.standardize(y2)
        # print('y shape = ', y.shape)
        y = torch.sum(y * weight, axis=1)
        results = {self.property: y,'weight':weight}

        return results


def softmax(x,mask):
    exp_input = torch.exp(x)

    # Set the contributions of "masked" atoms to zero
    if mask is not None:
            # If the mask is lower dimensional than the array being masked,
            #  inject an extra dimension to the end
        if mask.dim() < x.dim():
            mask = torch.unsqueeze(mask, -1)
        exp_input = torch.where(mask > 0, exp_input, torch.zeros_like(exp_input))

        # Sum exponentials along the desired axis
    exp_input_sum = torch.sum(exp_input, axis=1, keepdim=True)

        # Normalize the exponential array by the
    weights = exp_input / exp_input_sum
    return weights
