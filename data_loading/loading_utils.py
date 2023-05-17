import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def select_target_id(ds_geometric, target_id):
    geom_data = []
    for data in ds_geometric:
        data_obj = Data(
            z=data.z,
            pos=data.pos,
            y=data.y[target_id].reshape(1,) if data.y.dim() == 1 else data.y[:, target_id].reshape(1,),
            num_nodes=data.num_nodes
        )
        if hasattr(data, 'y_names'):
            data_obj.y_names = [data.y_names[target_id]]
        geom_data.append(data_obj)

    return geom_data


def train_scaler(ds_geometric):
    ys = np.array([data.y.item() for data in ds_geometric]).reshape(-1, 1)
    scaler = StandardScaler()
    scaler = scaler.fit(ys)

    return scaler


def scale_dataset(ds_geometric, scaler):
    ys = np.array([data.y.item() for data in ds_geometric]).reshape(-1, 1)
    ys_scaled = scaler.transform(ys)

    geom_data = []
    for idx, data in enumerate(ds_geometric):
        data_obj = Data(
            z=data.z,
            pos=data.pos,
            y=torch.tensor(ys_scaled[idx]),
            num_nodes=data.num_nodes
        )
        if hasattr(data, 'formula'):
            data_obj.formula = data.formula
        if hasattr(data, 'y_names'):
            data_obj.y_names = data.y_names

        geom_data.append(data_obj)

    return geom_data