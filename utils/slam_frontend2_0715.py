import time
import random
import numpy as np
import gc
import torch
import torch.multiprocessing as mp
import cv2
from tqdm import tqdm
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians, eval_rendering
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth , get_median_depth_wo_opacity
from utils.submap_utils import Submap


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config        
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.submap_list =[]
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.active_submap = None

        self.reset = True
        self.requested_init = False
        self.requested_new_submap = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.last_kf = 0
        self.cameras = dict()      
        self.device = "cuda:0"
        self.pause = False

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.max_kf_size = self.config["Training"]["max_kf_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, viewpoint,depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if(init):
            self.kf_indices.append(cur_frame_idx)
        # viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                
                depth = depth.detach().clone()
                if (init) :
                    opacity = opacity.detach()
                    use_inv_depth = False
                    if use_inv_depth:
                        inv_depth = 1.0 / depth
                        inv_median_depth, inv_std, valid_mask = get_median_depth(
                            inv_depth, opacity, mask=valid_rgb, return_std=True
                        )
                        invalid_depth_mask = torch.logical_or(
                            inv_depth > inv_median_depth + inv_std,
                            inv_depth < inv_median_depth - inv_std,
                        )
                        invalid_depth_mask = torch.logical_or(
                            invalid_depth_mask, ~valid_mask
                        )
                        inv_depth[invalid_depth_mask] = inv_median_depth
                        inv_initial_depth = inv_depth + torch.randn_like(
                            inv_depth
                        ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                        initial_depth = 1.0 / inv_initial_depth
                    else:
                        median_depth, std, valid_mask = get_median_depth(
                            depth, opacity, mask=valid_rgb, return_std=True
                        )
                        invalid_depth_mask = torch.logical_or(
                            depth > median_depth + std, depth < median_depth - std
                        )
                        invalid_depth_mask = torch.logical_or(
                            invalid_depth_mask, ~valid_mask
                        )
                        depth[invalid_depth_mask] = median_depth
                        initial_depth = depth + torch.randn_like(depth) * torch.where(
                            invalid_depth_mask, std * 0.5, std * 0.2
                        )
                else :
                    median_depth, std, valid_mask = get_median_depth_wo_opacity(
                            depth, mask=valid_rgb, return_std=True
                        )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                        )                   

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        print("depth")
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        
        # remove everything from the queues - backend queue 비우기
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        GT = torch.eye(4)
        R_ = torch.eye(3)
        T_ = GT[3,:3]        
        viewpoint.update_RT(R_, T_)
        # print(viewpoint.R_gt)
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        depth_map = self.add_new_keyframe(cur_frame_idx, viewpoint, init=True)
        
        # imname = "depth.png"
        # cv2.imwrite(imname,depth_map*10)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False
    
    def reset_current_submap(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular 
        self.iteration_count = 0       
       
        # remove everything from the queues - backend queue 비우기
        while not self.backend_queue.empty():
            self.backend_queue.get()
        GT = torch.eye(4)
        R_ = torch.eye(3)
        T_ = GT[3,:3]        
        viewpoint.update_RT(R_, T_)
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)    
        depth_map = self.add_new_keyframe(cur_frame_idx, viewpoint, init=True)
        self.reqeust_new_submap(cur_frame_idx,viewpoint,depth_map,True)
        for i in  self.active_submap.kf_idx[:-1]:
            if i in self.kf_indices:
                self.kf_indices.remove(i)
                self.cleanup(i)
        
        # self.active_submap = None
    
    
    def tracking(self, cur_frame_idx, viewpoint):
        # print('track!')
   
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)
   
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )        
        pose_optimizer = torch.optim.Adam(opt_params)
        # if not self.initialized:
        #     self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]*3
        # else :
        #     self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        for tracking_itr in range(self.tracking_itr_num):
            # print("before len gaussian = %i" %len(self.active_submap.gaussians._xyz))
            render_pkg = render(
                viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            # print("after len gaussian = %i" %len(self.active_submap.gaussians._xyz))
            # print(" ")
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
        
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
       
            if converged:       
                # print("[converge]trcaking_itr =  %i" %tracking_itr)   
                break          
        # print("[finish]trcaking_itr =  %i" %tracking_itr)   
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg
      
    
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):  
        # if cur_frame_idx % 8 == 0 :#  or  cur_frame_idx == 716 or cur_frame_idx == 717:
        #     print("key fraaaaame~!")
        #     return True
        # else :
        #     return False
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame =self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        # last_kf = self.active_submap.viewpoints[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check
    
    def is_new_submap(
        self
        # cur_frame_idx,
        # last_keyframe_idx,
        # cur_frame_visibility_filter,
        # occ_aware_visibility,
        # viewpoint,
        # depth_map
    ):  
                      
        # if(cur_frame_idx == 720 or cur_frame_idx == 1048 or cur_frame_idx == 1496):
        #     return True  
        
        # if(cur_frame_idx % 352 ==0 ):
            # return True      
        # if (self.active_submap.get_submap_size()>=self.max_kf_size):
        #     print("active map size exceed max kf size %i , create new sub map!"%self.active_submap.get_submap_size())
        #     return True       

        # kf_translation = self.config["Training"]["kf_translation"]
        # kf_min_translation = self.config["Training"]["kf_min_translation"]
        # kf_overlap = self.config["Training"]["kf_overlap"]

        # curr_frame = self.cameras[cur_frame_idx]
        # last_kf = self.active_submap.viewpoints[last_keyframe_idx]
        # pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        # last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        # last_kf_WC = torch.linalg.inv(last_kf_CW)
        # dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # dist_check = dist > kf_translation * self.median_depth
        # dist_check2 = dist > kf_min_translation * self.median_depth

        # union = torch.logical_or(
        #     cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        # ).count_nonzero()
        # intersection = torch.logical_and(
        #     cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        # ).count_nonzero()
        # point_ratio_2 = intersection / union
        # if(point_ratio_2 < kf_overlap and dist_check2):
        #     print("excessive pose jump happen!, create new sub map!")
        #     # self.requested_new_submap(cur_frame_idx,viewpoint,depth_map)
        #     return True      
        return  False
    
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = window + [cur_frame_idx]
        # remove frames which has little overlap with the current frame
        curr_frame = self.active_submap.viewpoints[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(len(window)- N_dont_touch):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            # print(cur_frame_visibility_filter.shape)
            # print(self.occ_aware_visibility[kf_idx].shape)
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            print("length of to_remove = %i" %len(to_remove))
            print("remove %i frame from window" %to_remove[0])
            window.remove(to_remove[0])
            removed_frame = to_remove[0]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range( len(window)-N_dont_touch):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.active_submap.viewpoints[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(len(window)-N_dont_touch):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.active_submap.viewpoints[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[idx]
            window.remove(removed_frame)

        return window, removed_frame
    

    def add_to_submap(
        self, cur_frame_idx, occ_aware_visibility, viewpoint ):  
        self.active_submap.viewpoints[cur_frame_idx] = viewpoint
        # self.active_submap.occ_aware_visibility[cur_frame_idx] = occ_aware_visibility
        self.active_submap.kf_idx.append(cur_frame_idx)
        
        return False
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap,kf_idx):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap,kf_idx]
        self.backend_queue.put(msg)
        #self.last_kf = cur_frame_idx
        # print("[f-k]last kf = %i" % self.last_kf)
        # print("f  : tag = keyframe")
        self.requested_keyframe += 1

    def reqeust_new_submap(self, cur_frame_idx, viewpoint,depth_map, reset_ = False):
       
        self.last_kf = self.active_submap.get_last_frame_idx()
        if not(reset_):
            self.submap_list.append(self.active_submap)
     
        msg = ["new_map", cur_frame_idx,viewpoint,depth_map]
        # msg = ["new_map", cur_frame_idx,viewpoint,last_kf,depth_map]
        self.backend_queue.put(msg)
        print("[f-n]last kf = %i" % self.last_kf)
        # print("f  : tag = new map")
        self.requested_new_submap = True
      

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        # print("f  : tag = init")
        self.requested_init = True

    def sync_backend(self, data):
   
        if data[0] == "init" or data[0] =="new_map":
            self.active_submap = data[1]    
            self.occ_aware_visibility = self.active_submap.occ_aware_visibility 
            # print("sync_backend!")        
            self.last_kf = self.active_submap.current_window[-1] 
            for kf_id in self.active_submap.current_window:
                kf = self.active_submap.viewpoints[kf_id]           
                self.cameras[kf_id].update_RT(kf.R.clone(), kf.T.clone())      
            
        else :
            self.active_submap.gaussians = data[1]    
            self.occ_aware_visibility = data[2]        
            keyframes = data[3]
            self.last_kf = self.active_submap.current_window[-1]            
        
            for kf_id, kf_R, kf_T in keyframes:
                self.active_submap.viewpoints[kf_id].update_RT(kf_R.clone(), kf_T.clone())         
                self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
    
    def sync_backend2(self, data):
        self.active_submap = data[1]    
        self.last_kf = self.active_submap.current_window[-1]
        self.occ_aware_visibility = self.active_submap.occ_aware_visibility
        # print("sync_backend!")        
        for kf_id in self.active_submap.current_window:
            kf = self.active_submap.viewpoints[kf_id]           
            self.cameras[kf_id].update_RT(kf.R.clone(), kf.T.clone())      

    def cleanup(self, cur_frame_idx): # R,T는 놔두고 나머지만 지움
        self.cameras[cur_frame_idx].clean()        
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            
            #print("current_window %i "%len(self.current_window))
            # visualize관련 queue 부분
            # if self.q_vis2main.empty():
            #     if self.pause:
            #         continue
            # else:
            #     data_vis2main = self.q_vis2main.get()
            #     self.pause = data_vis2main.flag_pause
            #     if self.pause:
            #         self.backend_queue.put(["pause"])
            #         continue
            #     else:
            #         self.backend_queue.put(["unpause"])
                    
            #여기부터 진짜 시작
            if self.frontend_queue.empty():
                tic.record()
                # print("1")
                #모든 데이터 다보면 save
                if cur_frame_idx >= len(self.dataset):
                    self.submap_list.append(self.active_submap)
                    if self.save_results:
                        eval_ate(
                            self.submap_list,
                            self.active_submap,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        # total_psnr = 0
                        # total_ssim = 0
                        # total_lpips = 0
                        
                        # rendering_result = eval_rendering(
                        # self.cameras,
                        # # anchor_frame_matrix,
                        # self.active_submap.gaussians,
                        # self.dataset,
                        # self.save_dir,
                        # self.pipeline_params,
                        # self.active_submap.background,
                        # kf_indices=self.active_submap.kf_idx,
                        # iteration="before_opt",
                        # )
                        # total_psnr+=rendering_result["mean_psnr"]
                        # total_ssim +=rendering_result["mean_ssim"]
                        # total_lpips +=rendering_result["mean_lpips"]  
                        # print("PSNR = %f" %total_psnr)
                        # save_gaussians(
                        #     self.active_subamap.gaussians, self.save_dir, "final", final=True
                        # )
                    break
                
                #request_init 일경우 backend의 init -> continue
                if self.requested_init :
                    time.sleep(0.01)
                    continue
                if self.requested_new_submap:
                    time.sleep(0.01)
                    continue
                    
                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                # initialize안되었는데 keyframe backend 보낸이후
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
             
                # 최초 카메라 지정    
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )                
                viewpoint.compute_grad_mask(self.config)
                self.cameras[cur_frame_idx] = viewpoint               
              
                #시작할때, 혹은 reset이 필요할때 -> initialize 다시함
                if self.reset:
                    # print("4")
                    self.initialize(cur_frame_idx, viewpoint)
                    # self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue
                print("cur_frame_id = %i"%cur_frame_idx)        
                self.initialized = self.initialized or (
                    len(self.active_submap.current_window) == self.window_size
                )
                #tr1 = time.time()    
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                #tr2 = time.time()
                # print("tracking time = "+str(tr2-tr1))
                # current_window_dict = {}
                # current_window_dict[self.current_window[0]] = self.current_window[1:]
                # keyframes = [self.active_submap.viewpoint[kf_idx] for kf_idx in self.active_submap.current_window]                # print("request keyframe = %i" %self.requested_keyframe)
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue                
               
                last_keyframe_idx = self.last_kf
                # print("last_kf = %i" %last_keyframe_idx)
                # print("check = %i" %(cur_frame_idx - last_keyframe_idx))
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                # print(cur_frame_idx - last_keyframe_idx)
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                # print(curr_visibility.shape)
                # print(self.occ_aware_visibility[last_keyframe_idx].shape)
                # print("is cur_frame keyframe?")
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                # print("kf test = %r"%create_kf)
                # print("cur vis = %i, last kf = %i " %(curr_visibility.count_nonzero(),self.occ_aware_visibility[last_keyframe_idx].count_nonzero()))
                if len(self.active_submap.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    print("point_ratio = %f , check_time= %i" %(point_ratio,check_time))
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                
                # print("kf test2 = %r"%create_kf)
                # if self.single_thread:
                #     create_kf = check_time and create_kf
                if create_kf:
                    #print("cur frame is keyframe")
                    #self.kf_indices.append(cur_frame_idx)
                    # d1 = time.time()
                    
                    # d2= time.time()
                    # print("depth map time = "+str(d2-d1))
                    create_new_submap = self.is_new_submap()
                    #cur_frame_idx, last_keyframe_idx, curr_visibility,
                    #self.occ_aware_visibility,viewpoint,depth_map)
                    if(create_new_submap):
                        # depth_map = self.add_new_keyframe(cur_frame_idx, viewpoint,depth=render_pkg["depth"], init=False)
                        d1= time.time()
                        depth_map = self.add_new_keyframe(cur_frame_idx, viewpoint, init=True)
                        d2= time.time()
                        self.reqeust_new_submap(cur_frame_idx,viewpoint,depth_map)
                        print("[new map] depth map time = "+str(d2-d1))
                    else :
                        self.add_to_submap( cur_frame_idx,
                            curr_visibility,
                            # self.occ_aware_visibility,
                            self.cameras[cur_frame_idx]
                        )
                        w1= time.time()
                        self.active_submap.current_window, removed = self.add_to_window(
                            cur_frame_idx,
                            curr_visibility,
                            self.occ_aware_visibility,
                            self.active_submap.current_window,
                        )
                        w2 =time.time()
                        # print("window time = "+str(w2-w1))
                        if self.monocular and not self.initialized and removed is not None:
                            # self.reset = True
                            Log(
                                "Keyframes lacks sufficient overlap to initialize the map, resetting."
                            )
                            print("kf idx = ",end="")
                            for i  in self.kf_indices :
                                print(" %i"%i,end="")
                            print("")
                            self.reset_current_submap(cur_frame_idx, viewpoint)
                            cur_frame_idx += 1
                            continue    
                        depth_map = self.add_new_keyframe(
                            cur_frame_idx,
                            viewpoint,                        
                            depth=render_pkg["depth"],
                            opacity=render_pkg["opacity"],
                            init=True,
                        )                                      
                        self.request_keyframe(
                            cur_frame_idx, viewpoint, self.active_submap.current_window, depth_map, self.active_submap.kf_idx)
                    
                        print("kf idx = ",end="")
                        for i  in self.kf_indices :
                            print(" %i"%i,end="")
                        print(" [%i]" %len(self.kf_indices))       
                            
                        if (
                            self.save_results
                            and len(self.kf_indices) >=4 
                            and self.save_trj
                            and create_kf
                            and len(self.kf_indices) % self.save_trj_kf_intv == 0
                            ):
                            Log("Evaluating ATE at frame: ", cur_frame_idx)
                            # print("len of kf indices = %i "%len(self.kf_indices))
                            eval_ate(                        
                                self.submap_list,
                                self.active_submap,
                                self.save_dir,
                                cur_frame_idx,
                                False,
                                monocular=self.monocular,
                            )                 
                    # # print("last_kf_idx  = %i"%self.last_kf)
                    # # print("total submap num  = %i"%len(self.submap_list))
                    # print("active_submap size = %i , cur_win_size = %i"%(self.active_submap.get_submap_size(),self.active_submap.get_win_size()))
                    # print("active sub map idx [kf_idx] = ",end="")
                    # for i  in self.active_submap.kf_idx :
                    #     print(" %i"%i,end="")
                    # print(" ")
                    # # print("active sub map idx [viewpoints] = ",end="")
                    # # for idx, view in self.active_submap.viewpoints.items() :
                    # #     print(" %i"%idx,end="")
                    # # print(" ")
                    # print("current_window idx = ",end="")
                    # for i  in self.active_submap.current_window :
                    #     print(" %i"%i,end="")
                    # print(" ")    
                    
                else:
                    # print("cur frame is not keyframe!")
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1
                
                    # rendering_result = eval_rendering(
                    # self.cameras,
                    # # anchor_frame_matrix,
                    # self.active_submap.gaussians,
                    # self.dataset,
                    # self.save_dir,
                    # self.pipeline_params,
                    # self.active_submap.background,
                    # kf_indices=self.active_submap.kf_idx,
                    # iteration="before_opt",
                    # )
                toc.record()               
                torch.cuda.synchronize()               
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                # print("b num of queue = %i " %self.frontend_queue.qsize())
                data = None
                if (self.frontend_queue.qsize() ==1):
                    data = self.frontend_queue.get()    
                while self.frontend_queue.qsize() >=1 :
                    data = self.frontend_queue.get()
                    if data[0]=="sync_backend":
                        continue
                    else :
                        break                
                # print("a num of queue = %i, tag = %s " %(self.frontend_queue.qsize(),data[0]))
                
                # data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    #print("bf : tag = sync_backend")
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    #print("bf  : tag = keyframe")
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    #print("bf : tag = init")
                    self.sync_backend(data)
                    self.requested_init = False
                    
                elif data[0] == "new_map":
                    #print("bf : tag = new_map")
                    self.sync_backend(data)
                    self.requested_new_submap = False 
                    self.initialized = not self.monocular                                                                 
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
                
                while not self.frontend_queue.empty():
                    self.frontend_queue.get()
                
                
                
                   # print("5")
                # if not self.reset:
                #     print("---------before tracking-------")
                #     print("last_kf_idx  = %i"%self.last_kf)
                #     print("active_submap size = %i , cur_win_size = %i"%(self.active_submap.get_submap_size(),self.active_submap.get_win_size()))
                #     print("active sub map idx [kf_idx] = ",end="")
                #     for i  in self.active_submap.kf_idx :
                #         print(" %i"%i,end="")
                #     print(" ")
                #     print("active sub map idx [viewpoints] = ",end="")
                #     for idx, view in self.active_submap.viewpoints.items() :
                #         print(" %i"%idx,end="")
                #     print(" ")
                #     print("current_window idx = ",end="")
                #     for i  in self.active_submap.current_window :
                #         print(" %i"%i,end="")
                #     print(" ")  
                # print("--------------------------") 
                # Tracking