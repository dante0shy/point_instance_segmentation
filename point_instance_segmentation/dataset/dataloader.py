import json
import os
import numpy as np
import torch
from torch_geometric.data import Data,Batch
from MinkowskiEngine.utils import sparse_quantize,sparse_collate

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
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extras', 'val_npy.json'))
)
# batch_size = 4


# class PanopticLabels(NamedTuple):
#     center_label: torch.Tensor
#     y: torch.Tensor
#     num_instances: torch.Tensor
#     instance_labels: torch.Tensor
#     instance_mask: torch.Tensor
#     vote_label: torch.Tensor

class DataLoader():
    def __init__(self,data = train,batch_size = 4):
        self.feature_dimension = 1
        self.num_classes = 3
        self.stuff_classes = torch.Tensor([0])
        self.data_list = data
        self.data= [self.read(x) for x in self.data_list]
        self.grid_size =  0.05
        self.batch_size = batch_size

    def read(self,data):
        label = json.load(open(data[2]))
        with open(data[1], 'rb') as f:
            pcd = torch.Tensor(np.load(data[1])) # ,allow_pickle=False
        # print(pcd.shape)
        pcd = pcd

        r = torch.ones(pcd.shape[0]).view(-1,1)
        label_map = torch.zeros(pcd.shape[0]) #-1
        instance_labels = torch.zeros(pcd.shape[0])
        vote_label = torch.zeros_like(pcd)
        center_label = []
        for i,l in enumerate(label):
            l_num = label_set.index(l['obj_type'])
            index, center = self.transform_index(l['psr'],pcd)
            label_map[index] =l_num
            instance_labels[index] = i+1#[index]
            center_label.append(l_num)
            vote_label[index] = (pcd[index]  - center)#/0.05
        coords = (pcd-pcd.min(0)[0])/ 0.05
        coords = coords.int()
        # coords, r, sem_label = sparse_quantize(coords, feats=r.reshape(-1,1), labels=sem_label.astype(np.int32), ignore_label=0, quantization_size=scale)#ME.utils.
        ind = sparse_quantize(coords, feats=r.reshape(-1, 1), labels=label_map.int(), ignore_label=0,
                              return_index=True)
        sp_data = {}
        instance_labels = instance_labels[ind[0]].long()
        vote_label = vote_label[ind[0]]/ 0.05#[instance_labels]
      # ME.utils.quantization_size=scale,
        # ins_label = torch.Tensor(label_map.astype(np.int))[ind[0]]
        # center_label: torch.Tensor
        # y: torch.Tensor
        # num_instances: torch.Tensor
        # instance_labels: torch.Tensor
        # instance_mask: torch.Tensor
        # vote_label: torch.Tensor
        sp_data["pos"] = pcd[ind[0]]
        sp_data["coords"] = coords[ind[0]]#[ind[0]]
        sp_data["rgb"] = r
        sp_data["y"] = ind[1].type(torch.LongTensor)#label_map#
        sp_data["x"] = r[ind[0]]
        sp_data["instance_labels"] = instance_labels
        sp_data["center_label"] = torch.Tensor(center_label)
        sp_data["num_instances"] = torch.Tensor(len(label))
        sp_data["instance_mask"] = instance_labels > 0#[ind[0]].type(torch.LongTensor)
        sp_data["vote_label"] = vote_label#[ind[0]]
        # sp_data["instance_bboxes"] = torch.from_numpy(instance_bboxes)
        # sp_data =Data(**sp_data)
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
               * (coords[:, 2] <= z1 + np.abs(psr['scale']['z'])) * (coords[:, 2] >= z1 - np.abs(psr['scale']['z'])), torch.Tensor([[x1,y1,z1]])
    def get_batch(self,x):
        return Batch.from_data_list([self.data[y] for y in x])


    def get_loader(self):
        return torch.utils.data.DataLoader(
                range(len(self.data)), batch_size= self.batch_size,collate_fn= self.get_batch  , num_workers=5, shuffle=False)
    # def get_loader(self):
    #     return torch.utils.data.DataLoader(
    #             self.data, batch_size=batch_size, num_workers=5, shuffle=True)

if __name__=='__main__':
    t_loader = DataLoader(train)

    for d in t_loader.get_loader():
        print(d[0][0].shape[0])
