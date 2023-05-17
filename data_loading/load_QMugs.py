import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from data_loading.loading_utils import train_scaler, scale_dataset, select_target_id


TARGET_ID_TO_PROPERTY = {
    0: 'GFN2:TOTAL_ENERGY',
    1: 'GFN2:ATOMIC_ENERGY',
    2: 'GFN2:FORMATION_ENERGY',
    3: 'GFN2:TOTAL_ENTHALPY',
    4: 'GFN2:TOTAL_FREE_ENERGY',
    5: 'GFN2:HOMO_ENERGY',
    6: 'GFN2:LUMO_ENERGY',
    7: 'GFN2:HOMO_LUMO_GAP',
    8: 'GFN2:FERMI_LEVEL',
    9: 'GFN2:DISPERSION_COEFFICIENT_MOLECULAR',
    10: 'GFN2:POLARIZABILITY_MOLECULAR',
    11: 'DFT:TOTAL_ENERGY',
    12: 'DFT:ATOMIC_ENERGY',
    13: 'DFT:FORMATION_ENERGY',
    14: 'DFT:XC_ENERGY',
    15: 'DFT:NUCLEAR_REPULSION_ENERGY',
    16: 'DFT:ONE_ELECTRON_ENERGY',
    17: 'DFT:TWO_ELECTRON_ENERGY',
    18: 'DFT:HOMO_ENERGY',
    19: 'DFT:LUMO_ENERGY',
    20: 'DFT:HOMO_LUMO_GAP'
}


def load_QMugs(random_seed: int, target_property_id: int):
    assert random_seed in [242566, 710365, 828444, 170963, 744683]
    assert target_property_id in range(21)

    qmugs = torch.load('data/QMugs/QMugs_3D_torch_geometric.pt')
    qmugs = select_target_id(qmugs, target_property_id)

    train, val, test = torch.utils.data.random_split(
        qmugs,
        lengths=[int(len(qmugs) * 0.8), int(len(qmugs) * 0.1), int(len(qmugs) * 0.1) + 1],
        generator=torch.Generator().manual_seed(random_seed)
    )

    scaler = train_scaler(train)

    train_scaled = scale_dataset(train, scaler)
    val_scaled = scale_dataset(val, scaler)
    test_scaled = scale_dataset(test, scaler)

    return train_scaled, val_scaled, test_scaled, scaler
