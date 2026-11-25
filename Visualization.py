import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import torch
from torch_cluster import knn_graph

from data_utils.Data_Visualization import PartNormalDataset

pv.global_theme.allow_empty_mesh = True


TARGET_INDEX = 6

configs = [
    {"step": 34, "dataset_id": 1, "title": "Step 34 (1s)"},
    {"step": 100, "dataset_id": 3, "title": "Step 100 (3s)"},
    {"step": 167, "dataset_id": 5, "title": "Step 167 (5s)"}
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m1_name = "Mean-Fluctuating Decomposition"
m1_path = "result/ournew/check/best_model.pth"


csv_path = "./rigid/single-leg/1.csv"

stl_q1_path = "./rigid/single-leg/q1.stl"
stl_q2_path = "./rigid/single-leg/q2.stl"

real_cm_in_cad1 = np.array([-57.963991, -75.924517, 39.963991])
real_cm_in_cad2 = np.array([-21.067079, -6.4640561, 3.5397459])


def get_recurdyn_matrix_degrees(tx, ty, tz, psi_deg, theta_deg, phi_deg):
    psi = np.radians(psi_deg)
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    Rz_psi = np.array(
        [
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi), np.cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    Rx_theta = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    Rz_phi = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0, 0],
            [np.sin(phi), np.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    Rotation = Rz_psi @ Rx_theta @ Rz_phi
    Rotation[0, 3] = tx
    Rotation[1, 3] = ty
    Rotation[2, 3] = tz
    return Rotation



def _build_radius_graph(pos, k=9):
    N = pos.shape[0]
    device = pos.device
    dist_matrix = torch.cdist(pos, pos, p=2)
    _, indices = torch.topk(dist_matrix, k=k + 1, dim=-1, largest=False, sorted=True)
    knn_indices = indices[:, 1:]
    src = torch.arange(N, device=device).unsqueeze(-1).repeat(1, k).reshape(-1)
    dest = knn_indices.reshape(-1)
    edge_index = torch.stack([src, dest], dim=0).to(torch.int64)
    src_pos = pos[edge_index[0]]
    dest_pos = pos[edge_index[1]]
    rel_pos = dest_pos - src_pos
    dist = torch.norm(rel_pos, p=2, dim=-1, keepdim=True)
    return edge_index, rel_pos, dist


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e8
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def full_coverage_knn_grouping(nsample, n_centroids, xyz):
    B, N, _ = xyz.shape
    device = xyz.device
    fps_idx = farthest_point_sample(xyz, n_centroids)
    centroids_xyz = index_points(xyz, fps_idx)
    dists = square_distance(centroids_xyz, xyz)
    covered_mask = torch.zeros(B, N, device=device)
    group_idx = torch.zeros(B, n_centroids, nsample, dtype=torch.long, device=device)
    for c in range(n_centroids):
        current_dist = dists[:, c, :].clone()
        penalty = covered_mask * 1e8
        adjusted_dist = current_dist + penalty
        _, idx = torch.topk(adjusted_dist, nsample, dim=-1, largest=False, sorted=False)
        group_idx[:, c, :] = idx
        covered_mask.scatter_(1, idx, 1.0)
    return group_idx


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)



def load_GNSmodel(name, path):
    mod1 = importlib.import_module(f"models.{name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1 = mod1.PositionAwareInteractionNet(
        velocity_dim=5,
        output_dim=3,
        num_node_types=3,
        hidden_dim=32,
        node_type_dim=4,
        n_message_passing=2
    ).to(device)
    state_dict = torch.load(path, map_location=device)
    net1.load_state_dict(state_dict)
    net1.eval()
    return net1


def load_model(name, path):
    mod1 = importlib.import_module(f"models.{name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1 = mod1.PositionAwareInteractionNet().to(device)
    net1.load_state_dict(torch.load(path))
    net1.eval()
    return net1


def gnnload_model(name, path):
    mod1 = importlib.import_module(f"models.{name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1 = mod1.PositionAwareInteractionNet().to(device)
    net1.load_state_dict(torch.load(path))
    net1.eval()
    return net1


def run_rollout_test(curr_particles_full, voxel, target_full_xyz, raw_jihe, boundary_features, model):
    group_idx = full_coverage_knn_grouping(50, 400, curr_particles_full)
    active_xyz = index_points(curr_particles_full, group_idx.reshape(1, -1)).view(1, 400, 50, 3)
    active_disp = index_points(voxel, group_idx.reshape(1, -1)).view(1, 400, 50, 3)
    active_target_world = index_points(curr_particles_full, group_idx.reshape(1, -1)).view(1, 400, 50, 3)

    for sub_step in range(step):
        jihe = raw_jihe[:, sub_step, :, :]
        centroids_xyz = torch.mean(active_xyz, dim=2)
        pred_m, pred_r, force, _ = model(active_xyz, active_disp, centroids_xyz, jihe, boundary_features)
        delta_disp = (pred_m.unsqueeze(-1) + pred_r).permute(0, 2, 3, 1)
        active_xyz = active_xyz + delta_disp
        active_disp = delta_disp

    return active_xyz, active_target_world, force, group_idx


def run_pointnet_test(curr_particles_full, voxel, target_full_xyz, raw_jihe, boundary_features, model):
    B, M, _ = curr_particles_full.shape
    active_xyz = curr_particles_full.view(B, 400, 50, 3).contiguous()
    active_disp = voxel.view(B, 400, 50, 3).contiguous()
    active_target_world = curr_particles_full.permute(0, 2, 1).contiguous()

    for sub_step in range(step):
        jihe = raw_jihe[:, sub_step, :, :]
        pred_r, pre_f = model(active_xyz, active_disp, jihe, boundary_features)
        pred_r = pred_r.view(B, 400, 50, 3)
        active_xyz = active_xyz + pred_r.view(B, 400, 50, 3)
        active_disp = pred_r

    return active_xyz.view(B, 400 * 50, 3), active_target_world.permute(0, 2, 1)


def new_gnn_run_rollout_test(curr_particles_full, curr_v, target_full_xyz, raw_jihe, boundary_features, net2):
    active_xyz = curr_particles_full.reshape(1, -1, 3)
    active_disp = curr_v.reshape(1, -1, 3)
    device = curr_particles_full.device
    B, N, _ = active_xyz.shape

    batch_idx = torch.zeros(B * N, dtype=torch.long, device=device)

    for sub_step in range(step):
        pos_tensor = active_xyz.squeeze(0)  # [N, 3]
        edge_index = knn_graph(pos_tensor, k=9, loop=False)  # [2, E]
        src, dest = edge_index[0], edge_index[1]

        rel_pos = pos_tensor[dest] - pos_tensor[src]  # [E, 3]
        dist = torch.norm(rel_pos, p=2, dim=-1, keepdim=True)  # [E, 1]
        jihe = raw_jihe[:, sub_step, :, :]

        pred_r, _ = net2(active_xyz, active_disp, edge_index, rel_pos, dist, jihe, boundary_features, batch_idx)

        active_xyz = active_xyz + pred_r
        active_disp = pred_r

    return active_xyz, curr_particles_full.reshape(1, -1, 3)


def new_cnn_run_rollout_test(curr_particles_full, curr_v, target_full_xyz, raw_jihe, boundary_features, net3):
    active_xyz = curr_particles_full.reshape(1, -1, 3)
    active_disp = curr_v.reshape(1, -1, 3)
    device = curr_particles_full.device

    for sub_step in range(step):
        gnn_cur = active_xyz.reshape(1, -1, 3)
        gnn_vel = active_disp.reshape(1, -1, 3).to(device)

        pos_tensor = gnn_cur.squeeze(0)
        edge_index = knn_graph(pos_tensor, k=9, loop=False)

        src, dest = edge_index[0], edge_index[1]
        edge_index = edge_index.unsqueeze(0)

        rel_pos = (pos_tensor[dest] - pos_tensor[src])
        jihe = raw_jihe[:, sub_step, :, :]

        pred_r, f = net3(gnn_cur, gnn_vel, jihe, boundary_features, edge_index, rel_pos.unsqueeze(0))
        active_xyz = active_xyz + pred_r
        active_disp = pred_r

    return active_xyz, curr_particles_full.reshape(1, -1, 3)


def run_rollout_gns(curr_particles_full, voxel, target_full_xyz, raw_jihe, boundary_features, net5):
    p_curr = curr_particles_full.clone().reshape(-1, 3)
    p_vel = voxel.clone().reshape(-1, 3)
    N_fluid = p_curr.shape[0]
    device = p_curr.device

    for sub_step in range(step):
        jihe = raw_jihe[:, sub_step, :, :].squeeze(0).T
        g_curr = jihe[:, :3]
        g_vel = jihe[:, 3:6]
        N_geom = g_curr.shape[0]
        gravity_val = jihe[0, 6] / 9.81
        wetness_val = jihe[0, 7]
        b_curr = boundary_features.squeeze(0).T[:, :3]
        b_vel = boundary_features.squeeze(0).T[:, 3:6]
        N_bound = b_curr.shape[0]

        N_total = N_fluid + N_geom + N_bound

        val_pair = torch.tensor([gravity_val, wetness_val], dtype=torch.float32).unsqueeze(0)
        global_params = torch.tile(val_pair, (N_total, 1)).to(device)

        all_pos = torch.cat([p_curr, g_curr, b_curr], dim=0)
        all_vel = torch.cat([p_vel, g_vel, b_vel], dim=0)
        all_vel_env = torch.cat([all_vel, global_params], dim=1)

        node_type = torch.cat([
            torch.zeros(N_fluid, dtype=torch.int64),
            torch.ones(N_geom, dtype=torch.int64),
            torch.full((N_bound,), 2, dtype=torch.int64)
        ], dim=0).to(device)

        non_kinematic_mask = torch.cat([
            torch.ones((N_fluid, 1), dtype=torch.bool),
            torch.zeros((N_geom + N_bound, 1), dtype=torch.bool)
        ], dim=0)

        edge_index, rel_pos, dist = _build_radius_graph(all_pos, k=9)

        pred_all, f = net5(all_vel_env, node_type, edge_index, rel_pos, dist)

        fluid_mask = non_kinematic_mask.squeeze(-1)
        pred_acc = pred_all[fluid_mask]

        p_curr = p_curr + pred_acc
        p_vel = pred_acc

    return p_curr



net1 = load_model(m1_name, m1_path)


def to_device_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.float().to(device)
    return torch.from_numpy(data).float().to(device)


try:
    df_motion = pd.read_csv(csv_path, header=None, encoding="gbk", engine="python")
except UnicodeDecodeError:
    df_motion = pd.read_csv(csv_path, header=None, encoding="utf-8-sig", engine="python")

stl_mesh1 = pv.read(stl_q1_path)
stl_mesh2 = pv.read(stl_q2_path)
stl_mesh1.points = stl_mesh1.points - real_cm_in_cad1
stl_mesh2.points = stl_mesh2.points - real_cm_in_cad2


def make_poly(p, v):
    pd = pv.PolyData(p)
    pd['Val'] = v
    return pd


row_data = []

for row_idx, cfg in enumerate(configs):
    step_val = cfg["step"]
    ds_id = cfg["dataset_id"]

    test_dataset = PartNormalDataset(ds_id)
    data_batch = test_dataset[TARGET_INDEX]
    (current_xyz, voxel_xyz, next_xyz, chazhi, res, mean_xyz, centroid_xyz, geometry,  boundary_features) = data_batch

    curr_particles_full = to_device_tensor(current_xyz, device).unsqueeze(0).reshape(1, -1, 3)
    voxel = to_device_tensor(voxel_xyz, device).unsqueeze(0).reshape(1, -1, 3)
    raw_jihe = to_device_tensor(geometry, device).unsqueeze(0)
    boundary_features = to_device_tensor(boundary_features, device).unsqueeze(0)
    target_full_xyz = to_device_tensor(next_xyz, device).unsqueeze(0).reshape(1, -1, 3)


    global step
    step = step_val

    with torch.no_grad():

        pd_f1_tensor, td_f1_tensor, _, group_idx = run_rollout_test(curr_particles_full, voxel, target_full_xyz,
                                                                    raw_jihe, boundary_features, net1)

        B, N, _ = curr_particles_full.shape
        curr_x_np = to_numpy(current_xyz).reshape(-1, 3) * 160

        error_m1_flat = torch.norm(pd_f1_tensor - td_f1_tensor, dim=-1).view(-1)
        flat_idx = group_idx.view(-1)
        err_m1_tensor = torch.zeros(N, device=device)
        err_m1_tensor[flat_idx] = error_m1_flat
        err_m1 = to_numpy(err_m1_tensor) * 160



        next_gt_np = to_numpy(next_xyz).reshape(-1, 3) * 160
        gt_disp = next_gt_np - curr_x_np
        gt_mag = np.linalg.norm(gt_disp, axis=1)

    pos_gt = next_gt_np.reshape(-1, 3)

    q1_last = df_motion.iloc[step_val - 1, 0:6].values
    q2_last = df_motion.iloc[step_val - 1, 6:12].values
    mat1_last = get_recurdyn_matrix_degrees(*q1_last)
    mat2_last = get_recurdyn_matrix_degrees(*q2_last)

    v_min_init = float(np.percentile(gt_mag, 10))
    v_max_init = float(np.percentile(gt_mag, 90))
    max_data_val = float(np.max(gt_mag))

    row_data.append({
        "cfg": cfg,
        "pos_gt": pos_gt,
        "gt_mag": gt_mag,
        "err_m1": err_m1,
        "mat1_last": mat1_last,
        "mat2_last": mat2_last,
        "v_min": v_min_init,
        "v_max": v_max_init,
        "max_val": max_data_val
    })


plotter = pv.Plotter(shape=(3, 2), window_size=[1600, 1100], border=False)
plotter.set_background("white")
font_path = "C:/Windows/Fonts/simhei.ttf"

sbar_args = dict(
    title='(mm)',
    title_font_size=16,
    label_font_size=14,
    fmt="%.2f",
    position_x=0.25,
    position_y=0.05,
    width=0.5,
    height=0.08
)

global_vmin = min([info["v_min"] for info in row_data])
global_vmax = max([info["v_max"] for info in row_data])
global_range = [global_vmin, global_vmax]
global_max_val = max([info["max_val"] for info in row_data])

m_args_base = dict(
    scalars='Val',
    cmap='viridis',
    point_size=6,
    render_points_as_spheres=True,
    scalar_bar_args=sbar_args,
    clim=global_range
)

all_meshes = []

for row_idx, info in enumerate(row_data):
    title_prefix = info["cfg"]["title"]
    mat1_l = info["mat1_last"]
    mat2_l = info["mat2_last"]

    plotter.subplot(row_idx, 0)
    plotter.add_text(f"{title_prefix} - Ground Truth", font_size=15, font_file=font_path, color="black")
    m_gt = plotter.add_mesh(make_poly(info["pos_gt"], info["gt_mag"]), **m_args_base, show_scalar_bar=True)
    m_act_0_1 = plotter.add_mesh(stl_mesh1.copy(deep=True), color="cyan", show_edges=True, opacity=0.4)
    m_act_0_1.user_matrix = mat1_l
    m_act_0_2 = plotter.add_mesh(stl_mesh2.copy(deep=True), color="tomato", show_edges=True, opacity=0.4)
    m_act_0_2.user_matrix = mat2_l

    plotter.subplot(row_idx, 1)
    plotter.add_text(f"{title_prefix} - Ours", font_size=15, font_file=font_path, color="black")
    m_ours = plotter.add_mesh(make_poly(info["pos_gt"], info["err_m1"]), **m_args_base, show_scalar_bar=False)
    m_act_1_1 = plotter.add_mesh(stl_mesh1.copy(deep=True), color="cyan", show_edges=True, opacity=0.4)
    m_act_1_1.user_matrix = mat1_l
    m_act_1_2 = plotter.add_mesh(stl_mesh2.copy(deep=True), color="tomato", show_edges=True, opacity=0.4)
    m_act_1_2.user_matrix = mat2_l



    all_meshes.extend([m_gt, m_ours])


def set_global_vmin(val):
    global_range[0] = val
    for m in all_meshes:
        m.mapper.scalar_range = global_range
    plotter.render()


def set_global_vmax(val):
    global_range[1] = val
    for m in all_meshes:
        m.mapper.scalar_range = global_range
    plotter.render()


y_pos = 0.02

s_min = plotter.add_slider_widget(
    callback=set_global_vmin,
    rng=[0, 30],
    value=global_vmin,
    title="Global vmin",
    pointa=(0.05, y_pos),
    pointb=(0.42, y_pos),
    color="blue"
)

s_max = plotter.add_slider_widget(
    callback=set_global_vmax,
    rng=[0, 30],
    value=global_vmax,
    title="Global vmax",
    pointa=(0.52, y_pos),
    pointb=(0.89, y_pos),
    color="red"
)

sliders = [s_min, s_max]


hint_text_actor = None


def do_export_figure():
    global hint_text_actor

    if hint_text_actor is not None:
        plotter.remove_actor(hint_text_actor)
        hint_text_actor = None

    for s in sliders:
        s.SetEnabled(False)

    plotter.render()

    out_filename = "Publication_Ready_Figure_MultiStep.png"
    plotter.screenshot(out_filename, transparent_background=False, scale=1)


    for s in sliders:
        s.SetEnabled(True)

    plotter.subplot(0, 0)
    hint_text_actor = plotter.add_text(
        "  [Status: Captured!] Press 'S' again to reshoot.",
        position=(30, 30),
        font_size=10,
        color="green"
    )
    plotter.render()


plotter.subplot(0, 0)
hint_text_actor = plotter.add_text(
    "  <- Press 'S' on keyboard to Export Figure",
    position=(30, 30),
    font_size=10,
    color="black"
)

plotter.add_key_event("s", do_export_figure)

plotter.link_views()
plotter.show()
plotter.close()