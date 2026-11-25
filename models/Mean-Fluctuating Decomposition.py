import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ConvBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, 1)
        self.bn1 = nn.GroupNorm(1, outchannels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn1(self.conv1(x)))


class ConvBlock1D(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inchannels, outchannels, 1)
        self.bn1 = nn.GroupNorm(1, outchannels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn1(self.conv1(x)))


class GeometryEncoder(nn.Module):
    def __init__(self, in_channel=6, out_channel=32):
        super(GeometryEncoder, self).__init__()
        self.pre = nn.Sequential(
            ConvBlock1D(in_channel, out_channel),
            ConvBlock1D(out_channel, out_channel)
        )

    def forward(self, x):
        x = self.pre(x)
        return torch.max(x, 2)[0].unsqueeze(-1)


class StructuralResidualGNN(nn.Module):
    def __init__(self, p_dim=64, f_dim=64, out_dim=64):
        super(StructuralResidualGNN, self).__init__()

        self.out_con = nn.Sequential(
            nn.GroupNorm(1, out_dim),
            ConvBlock(out_dim, out_dim),
            ConvBlock(out_dim, out_dim),
            ConvBlock(out_dim, out_dim)
        )
        self.dir_head = nn.Conv2d(out_dim, 3, kernel_size=1)
        self.mag_head = nn.Conv2d(out_dim, 1, kernel_size=1)

    def forward(self, p_feat, f_feat):
        x = p_feat + f_feat.unsqueeze(-1)
        feat = self.out_con(x)

        res_dir = F.normalize(self.dir_head(feat), dim=1, p=2, eps=1e-8)
        res_mag = F.softplus(self.mag_head(feat))

        p_res = res_dir * res_mag
        return p_res



class AttentionForceHead(nn.Module):
    def __init__(self, in_dim=64 + 64 + 16, hidden_dim=128, num_feet=4):

        super(AttentionForceHead, self).__init__()
        self.num_feet = num_feet

        self.attn_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.force_head = nn.Linear(hidden_dim, num_feet * 3)
        self.torque_head = nn.Linear(hidden_dim, num_feet * 3)

    def forward(self, f_feat, env_feat):

        attn_scores = self.attn_conv(f_feat)
        attn_weights = F.softmax(attn_scores, dim=-1)
        p_attn_mean = torch.sum(f_feat * attn_weights, dim=-1)

        p_max = torch.max(f_feat, dim=-1)[0]

        global_feat = torch.cat([p_attn_mean, p_max, env_feat], dim=-1)

        feat = self.mlp(global_feat)
        pred_force = self.force_head(feat)
        pred_torque = self.torque_head(feat)

        out = torch.cat([pred_force, pred_torque], dim=-1)
        return out


class PositionAwareInteractionNet(nn.Module):
    def __init__(self, n_centroids=400, n_neighbors=50, noise_std_pos=0.01, noise_std_feat=0.005,
                 detach_force_grad=True):
        super(PositionAwareInteractionNet, self).__init__()
        self.out = 64
        self.n_centroids = n_centroids
        self.detach_force_grad = detach_force_grad

        self.noise_std_pos = noise_std_pos
        self.noise_std_feat = noise_std_feat

        self.rel_embed = nn.Conv2d(3, self.out, kernel_size=1)
        self.vox_embed = nn.Conv2d(3, self.out, kernel_size=1)
        self.embed_norm = nn.GroupNorm(1, self.out)

        self.granular_pre = nn.Sequential(
            ConvBlock(self.out * 2, self.out),
            ConvBlock(self.out, self.out)
        )

        self.geom_encoder = GeometryEncoder(6, self.out)

        self.env_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1),
            nn.GroupNorm(1, 16),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=1),
            nn.GroupNorm(1, 16),
            nn.GELU()
        )

        self.fusion_mlp = nn.Sequential(
            ConvBlock1D(2 * self.out + 16, self.out),
            ConvBlock1D(self.out, self.out),
            ConvBlock1D(self.out, self.out)
        )

        self.mean_feat_head = nn.Sequential(
            ConvBlock1D(self.out, self.out),
            ConvBlock1D(self.out, self.out)
        )
        self.mean_dir_conv = nn.Conv1d(self.out, 3, kernel_size=1)
        self.mean_mag_conv = nn.Conv1d(self.out, 1, kernel_size=1)

        self.residual_gnn = StructuralResidualGNN(64, 64, out_dim=64)

        self.force_head_single = AttentionForceHead(in_dim=64 + 64 + 16, hidden_dim=128, num_feet=1)
        self.force_head_four = AttentionForceHead(in_dim=64 + 64 + 16, hidden_dim=128, num_feet=4)

        self.pos_encoder = nn.Sequential(
            ConvBlock1D(3, self.out),
            ConvBlock1D(self.out, self.out)
        )

    def forward(self, grouped_xyz, voxel_xyz, centroids_xyz, geom_data, boundary_features, geometry_type="single",
                foot_centers=None):

        B = grouped_xyz.shape[0]

        g_xyz = grouped_xyz.permute(0, 3, 1, 2)
        v_xyz = voxel_xyz.permute(0, 3, 1, 2)
        c_exp = centroids_xyz.view(B, self.n_centroids, 1, 3).permute(0, 3, 1, 2)

        rel_xyz = g_xyz - c_exp

        rel_feat = F.gelu(self.embed_norm(self.rel_embed(rel_xyz)))
        vox_feat = F.gelu(self.embed_norm(self.vox_embed(v_xyz)))

        p_feat = self.granular_pre(torch.cat([rel_feat, vox_feat], dim=1))
        g_feat_mean = torch.mean(p_feat, dim=3)
        g_feat_max = torch.max(p_feat, dim=3)[0]
        g_feat = g_feat_mean + g_feat_max


        c_mean = torch.mean(centroids_xyz, dim=1, keepdim=True)
        c_t = (centroids_xyz - c_mean).permute(0, 2, 1)

        geom_env = geom_data[:, 6:, 0]
        geom_geo = geom_data[:, :6, :]

        geom_env_clone = geom_env
        geom_env_clone[:, 0] = geom_env_clone[:, 0] / 9.81

        geom_geo = torch.cat([geom_geo, boundary_features], dim=2)
        geo_encoded = self.geom_encoder(geom_geo).repeat(1, 1, self.n_centroids)

        geom_env_input = geom_env_clone.unsqueeze(-1)
        env_raw_feat = self.env_encoder(geom_env_input)
        env_encoded = env_raw_feat.repeat(1, 1, self.n_centroids)
        c_t_encoded = self.pos_encoder(c_t)

        g_feat_weighted = g_feat + c_t_encoded
        f_feat = self.fusion_mlp(
            torch.cat([g_feat_weighted, geo_encoded, env_encoded], dim=1)
        )

        m_feat = self.mean_feat_head(f_feat)
        p_mean_dir = F.normalize(self.mean_dir_conv(m_feat), dim=1, p=2, eps=1e-8)
        p_mean_mag = F.softplus(self.mean_mag_conv(m_feat))
        p_mean = p_mean_dir * p_mean_mag

        p_res = self.residual_gnn(p_feat, f_feat)

        f_feat_for_force = f_feat
        env_feat_flat = env_raw_feat.squeeze(-1)

        if geometry_type == 'single':
            p_force = self.force_head_single(f_feat_for_force, env_feat_flat)
        elif geometry_type == 'four':
            p_force = self.force_head_four(f_feat_for_force, env_feat_flat)



        return p_mean, p_res, p_force, centroids_xyz

