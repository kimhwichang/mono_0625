import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    return ape_stat

def evaluate_evo_(poses_gt, poses_est,plot_dir,label,monocular=False,tag_=""):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag=tag_)
    
    
    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    return ape_stat



def eval_ate(submap_list, sub_, save_dir, iterations, final=False, monocular=False,new_submap = False,tag=""):
    trj_data = dict()
    latest_frame_idx = sub_.kf_idx[-1] + 2 if final else sub_.kf_idx[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        # print(pose)
        return pose

    def gen_pose_matrix2(R, T,M):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        M= M.numpy()
        pose = pose@M
        return pose
    if tag == "before" :
        print("before!")
        new_submap = False       
    final_index = 0 
    if(len(submap_list)==0): 
        for kf_id in sub_.kf_idx:      
            kf = sub_.viewpoints[kf_id]
            pose_est = np.linalg.inv(gen_pose_matrix2(kf.R, kf.T,sub_.get_anchor_frame_pose()))
            # if(kf_id == sub_.kf_idx[0]):
            #     pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
            pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

            trj_id.append(kf_id)    
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)
    
    else  : 
        for submap_ in submap_list:
            for kf_id in submap_.kf_idx:          
        
                kf = submap_.viewpoints[kf_id]
                pose_est = np.linalg.inv(gen_pose_matrix2(kf.R, kf.T,submap_.get_anchor_frame_pose()))
                # if(kf_id == sub_.kf_idx[0]):
                #     pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
                pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

                trj_id.append(kf_id)
                trj_est.append(pose_est.tolist())
                trj_gt.append(pose_gt.tolist())

                trj_est_np.append(pose_est)
                trj_gt_np.append(pose_gt)

        if(not final and not new_submap):
            print("not final")
            for kf_id in sub_.kf_idx:          

                kf = sub_.viewpoints[kf_id]
                pose_est = np.linalg.inv(gen_pose_matrix2(kf.R, kf.T,sub_.get_anchor_frame_pose()))
                # if(kf_id == sub_.kf_idx[0]):
                #     pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
                pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

                trj_id.append(kf_id)
                trj_est.append(pose_est.tolist())
                trj_gt.append(pose_gt.tolist())

                trj_est_np.append(pose_est)
                trj_gt_np.append(pose_gt)
    # print("len trj = %i " %len(trj_est_np))
    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    pose_dir = os.path.join(save_dir,"pose")
    mkdir_p(plot_dir)
    mkdir_p(pose_dir)
    

    label_evo = "final" if final else tag+" "+"{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    with open(os.path.join(pose_dir,"pose_timeline.txt"),"a",encoding="utf-8") as f:
        f.write(str(iterations) +" : "+str(round(ate,6))+"\n")
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
            
    return ate

def eval_ate_(sub_, tag_="", save_dir = "", monocular=False):
   
    first_idx = sub_.kf_idx[0]
    last_idx = sub_.kf_idx[-1]
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        # print(pose)
        return pose

    def gen_pose_matrix2(R, T,M):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        M= M.numpy()
        pose = pose@M
        return pose    

    for kf_id in sub_.kf_idx:      
        kf = sub_.viewpoints[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix2(kf.R, kf.T,sub_.get_anchor_frame_pose()))
        # pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))
        trj_id.append(kf_id)    
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)    
   
    Log("Evaluating ATE from frame : ", first_idx , " ~ " , last_idx)
    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)
    label_evo = "{:04}".format(last_idx)
    ate = evaluate_evo_(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,   
        plot_dir = plot_dir, 
        label= label_evo,  
        monocular=monocular,
        tag_ = tag_
    )   
            
    return ate

def eval_ate2(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    pose_dir = os.path.join(save_dir,"pose")
    mkdir_p(plot_dir)
    mkdir_p(pose_dir)
    
    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    with open(os.path.join(pose_dir,"pose_timeline.txt"),"a",encoding="utf-8") as f:
        f.write(str(iterations) +" : "+str(round(ate,6))+"\n")
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate

def eval_rendering(
    frames,  
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 3
    img_pred, img_gt, saved_frame_idx = [], [], []
    begin_idx = kf_indices[0]
    end_idx = kf_indices[-1]-1 
    
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    # print("frame num = %i"%len(frames))
    
    print("gaussian num = %i " %(gaussians._xyz.shape[0]))
    print("begin = %i , end = %i, total kf num = %i "%(begin_idx,end_idx, len(kf_indices)))
    count = 0
    for idx in range(begin_idx, end_idx, interval):
        count+=1
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        # pose = torch.eye(4)
        # pose[0:3, 0:3] = frame.R.clone()
        # pose[0:3, 3] = frame.T.clone()
        # pose = pose@anchor_pose
        # frame.R = pose[0:3, 0:3].to("cuda")
        # frame.t = pose[0:3, 3].to("cuda")        
        gt_image, _, _ = dataset[idx]
        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)
        im = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        im2= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # print(iteration=="before_opt")
        if(iteration=="before_opt"):
            image_name = "/workspace/MonoGS/slam2/frame_%i.png"%idx 
        elif(iteration=="final"):
            image_name = "/workspace/MonoGS/final/frame_%i.png"%idx 
        else :
            image_name = "/workspace/MonoGS/slam/frame_%i.png"%idx 
        # print(image_name)
        cv2.imwrite(image_name,im2)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["total frame num"] = count
    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)
    name_ =str(end_idx)
    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "result.json"), "a", encoding="utf-8"),
        indent=4,
    )
    return output

def eval_rendering_(
    frames,
    gaussians,
    dataset,  
    pipe,
    background,
    kf_indices,
    tag_="",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = kf_indices[-1]
    begin_idx = kf_indices[0]
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(begin_idx, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag = tag_,
    )
    return output



def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def save_gaussians_(gaussians, name, submap_idx, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/sub_{}".format(str(submap_idx))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

