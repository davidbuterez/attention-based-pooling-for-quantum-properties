import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from data_loading.loading_utils import train_scaler, scale_dataset, select_target_id


def scale_energies_train(train, test):
    y_train, y_test = [], []
    for data in train:
        y_train.append(data.energy)
    y_train = torch.stack(y_train).squeeze()

    for data in test:
        y_test.append(data.energy)
    y_test = torch.stack(y_test).squeeze()

    scaler = StandardScaler()
    scaler = scaler.fit(y_train.detach().cpu().numpy().reshape(-1, 1))

    y_train_scaled = scaler.transform(y_train.detach().cpu().numpy().reshape(-1, 1)).squeeze()
    y_test_scaled = scaler.transform(y_test.detach().cpu().numpy().reshape(-1, 1)).squeeze()

    train_scaled = []
    for idx, data in enumerate(train):
        data_obj = Data(
            z=data.z,
            pos=data.pos,
            y=torch.tensor(y_train_scaled[idx]).reshape(1,),
            num_nodes=data.num_nodes
        )
        train_scaled.append(data_obj)

    test_scaled = []
    for idx, data in enumerate(test):
        data_obj = Data(
            z=data.z,
            pos=data.pos,
            y=torch.tensor(y_test_scaled[idx]).reshape(1,),
            num_nodes=data.num_nodes
        )
        test_scaled.append(data_obj)

    return train_scaled, test_scaled, scaler


def load_MD17(ds, download_dir: str):
    assert ds in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']

    if ds == 'benzene':
        ds_load_name = 'benzene CCSD(T)'
    elif ds == 'aspirin':
        ds_load_name = 'aspirin CCSD'
    elif ds == 'malonaldehyde':
        ds_load_name = 'malonaldehyde CCSD(T)'
    elif ds == 'ethanol':
        ds_load_name = 'ethanol CCSD(T)'
    elif ds == 'toluene':
        ds_load_name = 'toluene CCSD(T)'

    train = torch_geometric.datasets.MD17(root=download_dir, name=ds_load_name, train=True)
    test = torch_geometric.datasets.MD17(root=download_dir, name=ds_load_name, train=False)

    if ds == 'toluene':
        test_size = int(len(test) // 2) + 1
    else:
        test_size = int(len(test) // 2)

    train, test, scaler = scale_energies_train(train, test)
    val, test = torch.utils.data.random_split(
        test,
        lengths=[int(len(test) // 2), test_size],
        generator=torch.Generator().manual_seed(42)
    )

    return train, val, test, scaler
