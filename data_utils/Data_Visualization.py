import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PartNormalDataset(Dataset):
    def __init__(self,time):
        self.testpath = "./data/20000"
        self.time = str(time)+"s"
        self.testpath  = os.path.join(self.testpath,self.time)

        self.test_keli_path =  os.path.join(self.testpath,"par")
        self.test_jihe_path =  os.path.join(self.testpath,"rigid")

        self.file_list = sorted([f for f in os.listdir(self.test_keli_path) if f.endswith('.npz')],key=lambda x: int(x.split('.')[0]))

        self.boundary_features = self.generate_boundary_points()
    def __getitem__(self, index):

        keli_file_path = os.path.join(self.test_keli_path, self.file_list[index])
        test_jihe_path = os.path.join(self.test_jihe_path, self.file_list[index])
        keli = np.load(keli_file_path,allow_pickle=True)
        jihe = np.load(test_jihe_path,allow_pickle=True)

        return (
            torch.from_numpy(keli['current_xyz']/160.0),
            torch.from_numpy(keli['voxel_xyz']/160.0),
            torch.from_numpy(keli['next_xyz']/160.0),
            torch.from_numpy(keli['chazhi']/160.0),
            torch.from_numpy(keli['res']/160.0),
            torch.from_numpy(keli['mean_xyz']/160.0),
            torch.from_numpy(keli['centroid_xyz']/160.0),
            torch.from_numpy(jihe["geometry"]),
            torch.from_numpy(self.boundary_features),
            )





    def __len__(self):
        return len(self.file_list)

    def generate_boundary_points(self):
        x_min, x_max = -56.67, 56.67
        y_min, y_max = -160.0, -30.0
        z_min, z_max = -127.3, 127.3

        num_points_xz = 15
        num_points_y = 10
        boundary_points = []
        x_grid = np.linspace(x_min, x_max, num_points_xz)
        z_grid = np.linspace(z_min, z_max, num_points_xz)
        for x in x_grid:
            for z in z_grid:
                boundary_points.append([x, y_min, z])

        y_grid = np.linspace(y_min, y_max, num_points_y)
        for x in x_grid:
            for y in y_grid:
                boundary_points.append([x, y, z_min])
                boundary_points.append([x, y, z_max])

        for z in z_grid:
            for y in y_grid:
                boundary_points.append([x_min, y, z])
                boundary_points.append([x_max, y, z])

        boundary_points = np.unique(np.array(boundary_points, dtype=np.float32), axis=0)

        boundary_points = boundary_points / 160.0
        num_b_points = boundary_points.shape[0]
        boundary_velocity = np.zeros((num_b_points, 3), dtype=np.float32)
        boundary_features = np.concatenate([boundary_points, boundary_velocity], axis=1)
        return boundary_features.T




if __name__ == '__main__':
    f =PartNormalDataset(1)
    current_xyz,voxel_xyz,next_xyz,chazhi,res,mean_xyz,centroid_xyz,geometry,li,boundary_features=f[0]
    a=1
