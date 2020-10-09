import sys,os,tqdm,glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
import torch
import torch.optim as optim
from torch_points3d.utils import config
from torch_points3d.models.panoptic import pointgroup
from point_instance_segmentation.model import pointgroup
from torch_geometric.data import Data, Batch
# from torch_points3d.datasets.batch import SimpleBatch
from point_instance_segmentation.dataset import dataloader
from point_instance_segmentation.utils import evaler
# from torch_points3d.datasets.panoptic import scannet
from omegaconf import OmegaConf
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# option = OmegaConf.create(yaml.load(open('./config.yaml')))
option = OmegaConf.load('./config.yaml')
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'extras','pg')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
num_points = 500
num_classes = 10
input_nc = 3
max_epoch = 2000
if __name__=='__main__':
    train_loader = dataloader.DataLoader(dataloader.train)
    val_loader = dataloader.DataLoader(dataloader.val)
    #Batch(batch=[1000], pos=[1000, 3], x=[1000, 3])
    # option, model_type, dataset, modules
    # option.mo
    pointgroup = pointgroup.PointGroup(
        option.models.PointGroup, None, train_loader, None
    ).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(unet.parameters())
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pointgroup.parameters()))
    evaler = evaler.evaler
    times = 10
    snap = glob.glob(os.path.join(log_dir, 'net*.pth'))
    snap = list(sorted(snap, key=lambda x: int(x.split('-')[-1].split('.')[0])))
    if snap:
        print('Restore from ' + snap[-1])
        pointgroup.load_state_dict(torch.load(snap[-1]))
        epoch_s = int(snap[-1].split('/')[-1].split('.')[0].split('-')[-1])
        optimizer.load_state_dict(torch.load(snap[-1].replace('net','optim')))
    for epoch in range(max_epoch):
        pointgroup.train()
        # pointgroup.zero_grad()
        loss = 0
        print('train {}'.format(epoch))
        with tqdm.tqdm(total=len(train_loader.data_list) // dataloader.batch_size) as pbar:
            for i,d in enumerate(train_loader.get_loader()):
        # for d in tqdm.tqdm(data_loader.get_loader(),total=len(data_loader.data_list)//dataloader.batch_size):
                pointgroup.zero_grad()
                data = d #Batch.from_data_list(d)
                # data.instance_mask = data.instance_mask.cuda()
                data.vote_label = data.vote_label.cuda()
                pointgroup.set_input(data, "cuda")
                pointgroup.forward(epoch)#epoch
                # print(data_out)
                pointgroup.backward()
                out = pointgroup.output
                # loss.backward()
                optimizer.step()
                loss += pointgroup.loss.item()
                pbar.set_postfix(
                    {'T': '{0:1.5f}'.format(loss/(i+1))})  # train_loss / (i + 1)
                pbar.update(1)
        if not epoch%times:
            torch.save(pointgroup.state_dict(), os.path.join(log_dir,'net-%09d'%epoch+'.pth'))
            torch.save(optimizer.state_dict(), os.path.join(log_dir,'optim-%09d'%epoch+'.pth'))
        # if not epoch % times:
            print('test {}'.format(epoch))
            pointgroup.eval()
            torch.cuda.empty_cache()
            evaler.reset()
            with tqdm.tqdm(total=len(val_loader.data_list) // dataloader.batch_size) as pbar:
                for i, d in enumerate(val_loader.get_loader()):
                    pointgroup.zero_grad()
                    data = d #Batch.from_data_list(d)
                    data.vote_label = data.vote_label.cuda()
                    pointgroup.set_input(data, "cuda")
                    pointgroup.forward()
                    out = pointgroup.output
                    prediction_semantic = out.semantic_logits.argmax(1)
                    evaler.addBatch(prediction_semantic.cpu().numpy(), d.y.cpu().numpy())
                    loss += pointgroup.loss.item()
                    pbar.set_postfix(
                        {'T': '{0:1.5f}'.format(evaler.getIoU()[0])})  # train_loss / (i + 1)
                    pbar.update(1)
            m_iou, iou = evaler.getIoU()
            acc= evaler.getacc()
            print('mean IOU', m_iou)
            print('mean acc', acc)
            tp, fp, fn = evaler.getStats()
            total = tp + fp + fn
            print('classes          IoU')
            print('----------------------------')
            for i in range(3):
                label_name = '{}'.format(i)
                print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, iou[i],
                                                                           tp[i],
                                                                           total[i]))
