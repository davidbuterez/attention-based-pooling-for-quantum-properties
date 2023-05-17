import numpy as np
import torch
from torch_geometric.data import Data

from data_loading.loading_utils import train_scaler, scale_dataset, select_target_id


TARGET_ID_TO_PROPERTY = {
    0: 'E1-CC2',
    1: 'E2-CC2',
    2: 'f1-CC2',
    3: 'f2-CC2',
    4: 'E1-PBE0/def2SVP',
    5: 'E2-PBE0/def2SVP',
    6: 'f1-PBE0/def2SVP',
    7: 'f2-PBE0/def2SVP',
    8: 'E1-PBE0/def2TZVP',
    9: 'E2-PBE0/def2TZVP',
    10: 'f1-PBE0/def2TZVP',
    11: 'f2-PBE0/def2TZVP',
    12: 'E1-CAM',
    13: 'E2-CAM',
    14: 'f1-CAM',
    15: 'f2-CAM'
}


def load_QM8(random_seed: int, target_property_id: int):
    assert random_seed in [291305, 259598, 999783, 490681, 159938]
    assert target_property_id in range(16)

    data = torch.load('data/QM8/QM8_3D_torch_geometric.pt')
    data = select_target_id(data, target_property_id)

    train, val, test = torch.utils.data.random_split(
        data,
        lengths=[int(len(data) * 0.8), int(len(data) * 0.1) + 1, int(len(data) * 0.1) + 1],
        generator=torch.Generator().manual_seed(random_seed)
    )

    scaler = train_scaler(train)

    train_scaled = scale_dataset(train, scaler)
    val_scaled = scale_dataset(val, scaler)
    test_scaled = scale_dataset(test, scaler)

    return train_scaled, val_scaled, test_scaled, scaler