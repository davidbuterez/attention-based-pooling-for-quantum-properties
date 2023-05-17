import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from data_loading.loading_utils import train_scaler, scale_dataset, select_target_id


TARGET_ID_TO_PROPERTY = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


def load_QM9(random_seed: int, target_property_id: int, download_dir: str):
    assert random_seed in [844249, 787755, 420455, 700990, 791796]
    assert target_property_id in range(12)

    qm9 = torch_geometric.datasets.QM9(root=download_dir)
    qm9 = select_target_id(qm9, target_property_id)

    train, val, test = torch.utils.data.random_split(
        qm9,
        lengths=[int(len(qm9) * 0.8), int(len(qm9) * 0.1), int(len(qm9) * 0.1) + 1],
        generator=torch.Generator().manual_seed(random_seed)
    )

    scaler = train_scaler(train)

    train_scaled = scale_dataset(train, scaler)
    val_scaled = scale_dataset(val, scaler)
    test_scaled = scale_dataset(test, scaler)

    return train_scaled, val_scaled, test_scaled, scaler