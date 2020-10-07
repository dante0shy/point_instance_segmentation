import json
import os
import numpy as np
import torch
from torch_geometric.data import Data
'''
(26097, 3)
[{'obj_id': '',
  'obj_type': 'Motor',
  'psr': {'position': {'x': 2.395234065998412,
                       'y': -0.758098281888457,
                       'z': -0.31995933316648006},
          'rotation': {'x': 0, 'y': 0, 'z': 2.9059732045705586},
          'scale': {'x': 0.09022762103496557,
                    'y': 0.6053844130294501,
                    'z': 0.7287165112793446}}},
 {'obj_id': '',
  'obj_type': 'Motor',
  'psr': {'position': {'x': 4.986832484198995,
                       'y': 0.3938771243629629,
                       'z': -0.5883759334683418},
          'rotation': {'x': 0, 'y': 0, 'z': 2.853613327010729},
          'scale': {'x': 0.4613079590092042,
                    'y': 0.6830779925260955,
                    'z': 0.6824910491704941}}}]
{'Pedestrian', 'Unknown', 'Motor'}
'''
label_set = ['Unknown','Pedestrian', 'Motor']

train = json.load(
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extras', 'train_npy.json'))
)
val = json.load(
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extras', 'train_npy.json'))
)
batch_size = 4


# class PanopticLabels(NamedTuple):
#     center_label: torch.Tensor
#     y: torch.Tensor
#     num_instances: torch.Tensor
#     instance_labels: torch.Tensor
#     instance_mask: torch.Tensor
#     vote_label: torch.Tensor

class DataLoader():
    feature_dimension = 1
    num_classes = 3
    stuff_classes = 5

    def __init__(self,data = train):
        self.data_list = train
        self.data= [self.read(x) for x in self.data_list]

    def read(self,data):
        label = json.load(open(data[2]))
        with open(data[1], 'rb') as f:
            pcd = torch.Tensor(np.load(data[1]))  # ,allow_pickle=False
        # print(pcd.shape)
        r = torch.ones(pcd.shape[0]).view(-1,1)
        label_map = torch.zeros(pcd.shape[0])
        instance_labels = torch.zeros(pcd.shape[0])

        for i,l in enumerate(label):
            l_num = label_set.index(l['obj_type'])
            index = self.transform_index(l['psr'],pcd)
            label_map[index] +=l_num
            instance_labels += (i+1)
        sp_data = {}

        sp_data["pos"] = pcd
        sp_data["rgb"] = r
        sp_data["y"] = label_map
        sp_data["x"] = pcd
        sp_data["instance_labels"] = instance_labels
        # sp_data["instance_bboxes"] = torch.from_numpy(instance_bboxes)
        sp_data =Data(**sp_data)

        return sp_data

    def transform_index(self,psr, coords):
        ct = np.cos(psr['rotation']['z'])
        st = np.sin(psr['rotation']['z'])
        x1 = psr['position']['x']
        y1 = psr['position']['y']
        z1 = psr['position']['z']
        tmp = torch.matmul(torch.Tensor([[ct, -st], [st, ct]]), (coords[:, :2] - torch.Tensor([x1, y1])).T).T

        return (tmp[:, 0] <= np.abs(psr['scale']['x'])) * (tmp[:, 1] <= np.abs(psr['scale']['y'])) \
               * (tmp[:, 0] >= -np.abs(psr['scale']['x'])) * (tmp[:, 1] >= -np.abs(psr['scale']['y'])) \
               * (coords[:, 2] <= z1 + np.abs(psr['scale']['z'])) * (coords[:, 2] >= z1 - np.abs(psr['scale']['z']))


    def get_loader(self):
        return torch.utils.data.DataLoader(
                self.data, batch_size=batch_size, num_workers=5, shuffle=True)
    # def get_loader(self):
    #     return torch.utils.data.DataLoader(
    #             self.data, batch_size=batch_size, num_workers=5, shuffle=True)

if __name__=='__main__':
    t_loader = DataLoader(train)

    for d in t_loader.get_loader():
        print(d[0][0].shape[0])
        print(d[0][1].shape[0])
        print(d[0][2].shape[0])