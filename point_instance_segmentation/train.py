import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
import torch
from torch_points3d.utils import config
from torch_points3d.models.panoptic import pointgroup
from torch_geometric.data import Data, Batch
# from torch_points3d.datasets.batch import SimpleBatch
from point_instance_segmentation.dataset import dataloader
# from torch_points3d.datasets.panoptic import scannet
from omegaconf import OmegaConf
import yaml
# option = OmegaConf.create(yaml.load(open('./config.yaml')))
option = OmegaConf.load('./config.yaml')
num_points = 500
num_classes = 10
input_nc = 3
max_epoch = 40
if __name__=='__main__':
    data_loader = dataloader.DataLoader(dataloader.train)
    #Batch(batch=[1000], pos=[1000, 3], x=[1000, 3])
    # option, model_type, dataset, modules
    pointgroup = pointgroup.PointGroup(
        option, None, dataloader, None
    )
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(unet.parameters())
    optimizer =  torch.optim.Adam(filter(lambda p: p.requires_grad, pointgroup.parameters()))


    for epoch in range(max_epoch):
        pointgroup.train()
        for d in data_loader.get_loader():
            pointgroup.zero_grad()
            data = Batch.from_data_list(d)
            pointgroup.set_input(data, "cuda")
            data_out = pointgroup.forward()
            pass