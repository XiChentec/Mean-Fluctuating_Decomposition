import argparse
import datetime
import gc
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils.读取npy_data import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Mean-Fluctuating Decomposition',
                        help='model name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate for finetuning')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-5, help='weight decay')

    return parser.parse_args()


def calculate_single_group_metrics(preds, gts, threshold=1e-3):
    preds = preds.flatten()
    gts = gts.flatten()

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))
    range_gts = np.max(gts) - np.min(gts)
    nrmse = (rmse / (range_gts + 1e-8)) * 100

    valid_mask = np.abs(gts) > threshold

    if np.sum(valid_mask) == 0:
        mape, mre = 0.0, 0.0
    else:
        valid_preds = preds[valid_mask]
        valid_gts = gts[valid_mask]

        sum_absolute_error = np.sum(np.abs(valid_gts - valid_preds))
        sum_absolute_gts = np.sum(np.abs(valid_gts))

        mape = (sum_absolute_error / (sum_absolute_gts + 1e-8)) * 100
        mre = sum_absolute_error / (sum_absolute_gts + 1e-8)

    if np.std(preds) < 1e-6 or np.std(gts) < 1e-6:
        pearson = 0.0
    else:
        pearson = np.corrcoef(preds, gts)[0, 1]

    return mae, rmse, mape, pearson, nrmse


def main(args):
    def log_string(str_msg):
        logger.info(str_msg)
        print(str_msg)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/part_seg')
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir.joinpath(args.log_dir if args.log_dir else timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('check/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(str(args))

    means = np.array([0.153812, 1.428451, -0.315181, 0.085468143, -0.026146426, 0.041413096], dtype=np.float32)
    stds = np.array([1.652391, 1.927874, 1.440526, 0.380076553, 0.052491013, 0.348902925], dtype=np.float32)

    log_string("Loading Data...")
    TRAIN_DATASET = PartNormalDataset(split='train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=6, drop_last=True, pin_memory=True,
                                                  persistent_workers=True, prefetch_factor=2)
    TEST_DATASET = PartNormalDataset(split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, drop_last=False, pin_memory=True)

    candidate_thresholds = {'Default': 1e-4, 'Active': 1e-2}
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.PositionAwareInteractionNet().float().cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-07,
                                 weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-8)

    best_score = 9999.0


    for epoch in range(args.epoch):
        log_string('\nEpoch %d / %d:' % (epoch + 1, args.epoch))

        classifier.train()
        train_accum = {}
        total_train_loss = 0.0
        for batch_id, data_batch in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), desc='Train'):
            curr_x, curr_v, _, chaz, rs, mn_x, ctr_x, jh_p, frc, boundary_features = [
                item.cuda(non_blocking=True) for item in data_batch]

            geometry_type = "single"
            optimizer.zero_grad()
            pred_mean, pred_residual, pred_force, _ = classifier(curr_x, curr_v, ctr_x, jh_p, boundary_features,
                                                                 geometry_type)
            loss, metrics, a = criterion(chaz, mn_x, rs, pred_mean, pred_residual, frc, pred_force, curr_x,
                                         geometry_type,
                                         thresh_dict=candidate_thresholds, metrics_butt="test", epoch=epoch,
                                         max_epoch=args.epoch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            for k, v in metrics.items():
                train_accum[k] = train_accum.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        count_gl = train_accum.get('Count_Global', 1e-8)
        mse_c_gl = train_accum.get('Sum_MSE_Coord_Global', 0) / count_gl

        log_string(f"Overall Metrics (Global):")
        log_string(f" Train/MSE : {mse_c_gl:.10f}")

        writer.add_scalar('Train/MSE', mse_c_gl, epoch)
        scheduler.step()

        classifier.eval()
        test_accum = {}
        all_force_preds, all_force_gts = [], []
        total_test_loss = 0.0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                for batch_id, data_batch in tqdm(enumerate(testDataLoader), total=len(testDataLoader), desc='Test'):
                    curr_x, curr_v, _, chaz, rs, mn_x, ctr_x, jh_p, frc, boundary_features = [
                        item.cuda(non_blocking=True) for item in data_batch]
                    pred_mean, pred_residual, pred_force, _ = classifier(curr_x, curr_v, ctr_x, jh_p, boundary_features,
                                                                         geometry_type)
                    loss, metrics, a = criterion(chaz, mn_x, rs, pred_mean, pred_residual, frc, pred_force, curr_x,
                                                 geometry_type,
                                                 thresh_dict=candidate_thresholds, metrics_butt="test", epoch=epoch,
                                                 max_epoch=args.epoch)
                    total_test_loss += loss.item()
                    for k, v in metrics.items():
                        test_accum[k] = test_accum.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

                    all_force_preds.append(pred_force.cpu().numpy())
                    all_force_gts.append(frc.cpu().numpy())

            log_string('=' * 85)

            count_gl = test_accum.get('Count_Global', 1e-8)
            mse_c_gl = test_accum.get('Sum_MSE_Coord_Global', 0) / count_gl
            mre_c_gl = test_accum.get('Sum_MRE_Coord_Num', 0) / (test_accum.get('Sum_MRE_Coord_Den', 0) + 1e-12)

            log_string(f"Overall Metrics (Global):")
            log_string(f" > MSE (Coord) : {mse_c_gl:.10f}")
            log_string(f" > MRE (Coord) : {mre_c_gl:.10f}")

            writer.add_scalar('Global/MSE_Coord', mse_c_gl, epoch)
            writer.add_scalar('Global/MRE_Coord', mre_c_gl, epoch)


            all_force_preds = np.concatenate(all_force_preds, axis=0) * stds + means
            all_force_gts = np.concatenate(all_force_gts, axis=0) * stds + means

            log_string("-" * 30 + " Force & Torque Metrics " + "-" * 30)
            log_string(
                f"{'Dim':<15} | {'MAE':<10} | {'NRMSE(%)':<10} ")


            log_string("-" * 96)

            pred_f_norm = np.linalg.norm(all_force_preds[:, :3], axis=1)
            gt_f_norm = np.linalg.norm(all_force_gts[:, :3], axis=1)
            f_norm_m = calculate_single_group_metrics(pred_f_norm, gt_f_norm)

            log_string(
                f"{'F_Norm (||F||)':<15} | {f_norm_m[0]:<10.4f} |  {f_norm_m[4]:<10.4f} ")
            writer.add_scalar('Force_Torque/F_Norm/MAE', f_norm_m[0], epoch)
            writer.add_scalar('Force_Torque/F_Norm/NRMSE', f_norm_m[4], epoch)

            pred_t_norm = np.linalg.norm(all_force_preds[:, 3:], axis=1)
            gt_t_norm = np.linalg.norm(all_force_gts[:, 3:], axis=1)
            t_norm_m = calculate_single_group_metrics(pred_t_norm, gt_t_norm)

            log_string(
                f"{'M_Norm (||M||)':<15} | {t_norm_m[0]:<10.4f} |  {t_norm_m[4]:<10.4f} ")
            writer.add_scalar('Force_Torque/M_Norm/MAE', t_norm_m[0], epoch)
            writer.add_scalar('Force_Torque/M_Norm/NRMSE', t_norm_m[4], epoch)

            log_string('=' * 85)

            # 保存模型逻辑
            current_score = mse_c_gl
            if current_score < best_score:
                best_score = current_score
                torch.save(classifier.state_dict(), str(checkpoints_dir) + '/best_model.pth')
                log_string(f" Saved Best Model (Score: {best_score:.10f})")



        torch.cuda.empty_cache()
        gc.collect()

    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)