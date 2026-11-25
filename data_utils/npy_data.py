import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PartNormalDataset(Dataset):
    def __init__(self,split='train',normal_channel=True):

        self.split = split
        self.normal_channel = normal_channel

        self.train_npy_path = "./data/train"

        self.test_npy_path = "./data/test"

        if self.split == 'train':
            self.data_path = self.train_npy_path

        elif self.split == 'test':
            self.data_path = self.test_npy_path




        self.file_list = sorted([f for f in os.listdir(self.data_path) if f.endswith('.npy')])

        self.boundary_features = self.generate_boundary_points()
        self.means = np.array([0.153812, 1.428451, -0.315181, 0.085468143, -0.026146426, 0.041413096], dtype=np.float32)
        self.stds = np.array([1.652391, 1.927874, 1.440526, 0.380076553, 0.052491013, 0.348902925], dtype=np.float32)

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

    def __getitem__(self, index):
        file_path = os.path.join(self.data_path, self.file_list[index])

        data = np.load(file_path, allow_pickle=True).item()
        raw_force = data['norm_force'].astype(np.float32)


        raw_force[3:6] = raw_force[3:6] / 1000.0



        norm_force_new = (raw_force - self.means) / self.stds
        force = torch.from_numpy(norm_force_new)



        return (
            torch.from_numpy(data['current_xyz']),
            torch.from_numpy(data['voxel_xyz']),
            torch.from_numpy(data['next_xyz']),
            torch.from_numpy(data['chazhi']),
            torch.from_numpy(data['res']),
            torch.from_numpy(data['mean_flow']),
            torch.from_numpy(data['centroid_xyz']),
            torch.from_numpy(data['geometry']),
            force,
            torch.from_numpy(self.boundary_features)
            )

    def __len__(self):
        return len(self.file_list)



if __name__ == '__main__':
    f =PartNormalDataset(split="train")
    current_xyz,voxel_xyz,next_xyz,chazhi,res,mean_flow,centroid_xyz,jihe_point,norm_force=f[1]
    a=1
