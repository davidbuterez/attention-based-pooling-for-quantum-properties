import numpy as np
import torch
from torch_geometric.data import Data

from data_loading.loading_utils import train_scaler, scale_dataset


TARGET_ID_TO_PROPERTY = {
    0: 'First excitation energy (ZINDO)',
    1: 'Electron affinity (ZINDO/s)',
    2: 'Excitation energy at maximal absorption (ZINDO)',
    3: 'Atomization energy (DFT/PBE0)',
    4: 'Highest occupied molecular orbital (GW)',
    5: 'Highest occupied molecular orbital (PBE0)',
    6: 'Highest occupied molecular orbital (ZINDO/s)',
    7: 'Maximal absorption intensity (ZINDO)',
    8: 'Ionization potential (ZINDO/s)',
    9: 'Lowest unoccupied molecular orbital (GW)',
    10: 'Lowest unoccupied molecular orbital (PBE0)',
    11: 'Lowest unoccupied molecular orbital (ZINDO/s)',
    12: 'Polarizability (self-consistent screening)',
    13: 'Polarizability (DFT/PBE0)'
}


def np_to_geometric_data(ds_array, target_id):
    geom_data = []
    for data in ds_array:
        data_obj = Data(
            formula=data[0],
            z=torch.from_numpy(data[1]).to(torch.long),
            pos=torch.from_numpy(data[2]).to(torch.float),
            y=torch.from_numpy(data[-2].astype(float)).to(torch.float)[target_id].reshape(1,),
            y_names=[data[4][target_id]],
            num_nodes=data[2].shape[0]
        )
        geom_data.append(data_obj)

    return geom_data


def load_QM7(random_seed: int, target_property_id: int):
    assert random_seed in [23887, 386333, 514094, 572909, 598587]
    assert target_property_id in range(14)

    train = np.load(f'data/QM7/QM7b_train_val_test_splits/{random_seed}/train.npy', allow_pickle=True)
    val = np.load(f'data/QM7/QM7b_train_val_test_splits/{random_seed}/validate.npy', allow_pickle=True)
    test = np.load(f'data/QM7/QM7b_train_val_test_splits/{random_seed}/test.npy', allow_pickle=True)

    train_geometric = np_to_geometric_data(train, target_property_id)
    val_geometric = np_to_geometric_data(val, target_property_id)
    test_geometric = np_to_geometric_data(test, target_property_id)

    scaler = train_scaler(train_geometric)

    train_scaled = scale_dataset(train_geometric, scaler)
    val_scaled = scale_dataset(val_geometric, scaler)
    test_scaled = scale_dataset(test_geometric, scaler)

    return train_scaled, val_scaled, test_scaled, scaler
