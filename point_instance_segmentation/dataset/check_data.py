import json,pprint
import os
import numpy as np
# from pypcd import pypcd
import struct
# import open3d as o3d
def read_pcd(filepath):
    lidar = []
    with open(filepath,'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)

def read_bin(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    return np_pcd

if __name__=='__main__':
    train = json.load(
        open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'extras','train_npy.json'))
    )
    labelset =[]
    for idx , data in enumerate(train):
        # pc = read_pcd(data[1])
        label = json.load(open(data[2]))

        labelset.extend([x['obj_type'] for x in label])
        if not idx:
            with open(data[1], 'rb') as f:
                pcd = np.load(data[1])  # ,allow_pickle=False
                print(pcd.shape)
            pprint.pprint(label)

    print(set(labelset))