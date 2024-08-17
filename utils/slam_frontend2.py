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
from gui import gui_utils, slam_gui
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, l1_loss_log_pow, l2_loss_log 
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians, eval_rendering_ ,eval_ate_
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth , get_median_depth_wo_opacity ,depth_reg   
from utils.submap_utils import Submap


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config        
        self.pipeline_params = None
        self.opt_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None  
        self.use_gui = False
        self.dataset = None
        self.initialized = False
        self.kf_indices = []
        self.submap_list =[]
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.active_submap = None   
        self.gui_process = None
 

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

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]        
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
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
        # print(self.cameras[cur_frame_idx].R)
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        
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
        # print(self.cameras[cur_frame_idx].R)
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)    
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_reset_submap(cur_frame_idx,viewpoint,depth_map)
        for i in  self.active_submap.kf_idx[:-1]:
            if i in self.kf_indices:
                self.kf_indices.remove(i)
                self.cleanup(i)
        
        # self.active_submap = None
    
    
    def pose_refine(self ,refine_iter):
        # print('track!')
        # print("i = %i" %(cur_frame_idx - self.use_every_n_frames))
        for idx in self.active_submap.kf_idx:
            viewpoint = self.active_submap.viewpoints[idx]
            viewpoint.reset_view_param()     
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
            
            for tracking_itr in range(refine_iter):  #self.tracking_itr_num):
                
                render_pkg = render(
                    viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )               
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
    
        return 0 
       
    
    def tracking(self, cur_frame_idx, viewpoint):
        # print('track!')
        # print("i = %i" %(cur_frame_idx - self.use_every_n_frames))
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
            pose_optimizer.zero_grad()
            # print("after len gaussian = %i" %len(self.active_submap.gaussians._xyz))
            # print(" ")         
            
            # gt_image = viewpoint.original_image.cuda()
            # _, h, w = gt_image.shape
            # mask_shape = (1, h, w)
            # rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
            # rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
            # rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
            # loss_i = (opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()
            # # loss_i = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))          
         
            # log_ = True
            # loss_d = 0
            # # log_ = False        
            # gt_depth = torch.Tensor(viewpoint.depth).unsqueeze(0).cuda()
            # depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
            # opacity_mask = (opacity > 0.95).view(*depth.shape)  
            # depth_mask = depth_pixel_mask * opacity_mask
            # depth = depth*depth_mask
            # gt_depth = gt_depth*depth_mask   
            # if log_ :
            #     gt_depth[torch.where(gt_depth<=1)] =1
            #     depth[torch.where(depth<=1)] =1             
            #     d1 = l1_loss_log_pow(depth,gt_depth)        
            #     d2 = l2_loss_log(depth, gt_depth)               
            #     loss_d = torch.sqrt(d2-0.85*d1)     
                
            #     # depth = torch.log(depth)
            #     # gt_depth = torch.log(gt_depth)          
            #     # dr = depth_reg(depth,gt_depth)           
            #     # ss = (1.0 - ssim(depth, gt_depth))      
            #     # loss_d =  0.6*d1 + 0.2* dr+ss           
            # else :
            #     d1 = l1_loss(depth,gt_depth) 
            #     loss_d =  0.6*d1 + 0.2* depth_reg(depth,gt_depth)+(1.0 - ssim(depth, gt_depth))
            # loss_tracking =loss_i*0.9+0.1*loss_d
            #------------------------------------------------------
            loss_tracking = get_loss_tracking( self.config, image, depth, opacity, viewpoint )                
            
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
        # if cur_frame_idx % 10 == 0 :#  or  cur_frame_idx == 716 or cur_frame_idx == 717:
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
        self,
        cur_frame_idx
    ):  
                      
        # if(cur_frame_idx == 720 or cur_frame_idx == 1048 or cur_frame_idx == 1496):
        #     return True  
        
        # if(cur_frame_idx % 352 ==0 ):
        #     return True      
        if (self.active_submap.get_submap_size()>=self.max_kf_size):
            print("active map size exceed max kf size %i , create new sub map!"%self.active_submap.get_submap_size())
            return True    
        return  False
    
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # window = window + [cur_frame_idx]       
        
        curr_frame = self.active_submap.viewpoints[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
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
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.active_submap.viewpoints[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
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
            removed_frame = window[N_dont_touch + idx]
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

    def check_mem(self,tag):
       
        print( tag+ f" Reserved Memory : {torch.cuda.memory_reserved()/1024.0/1024.0:.2f}MB" )
        print( tag+ f" Allocated Memory : {torch.cuda.memory_allocated()/1024.0/1024.0:.2f}MB")

    def request_new_submap(self, cur_frame_idx, viewpoint,depth_map):
        
        self.last_kf = self.active_submap.current_window[0]   
        self.eval_("sub_map finish")
        self.check_mem("before")
        self.submap_to_cpu(self.active_submap)    
        self.check_mem("after")
       
        self.submap_list.append(self.active_submap)         
        
        # tmp_t = torch.eye(4)
        # tmp_t[:3, :3]  = self.cameras[0].R_gt.clone()
        # tmp_t[:3, 3]   = self.cameras[0].T_gt.clone()        
        # tmp_inverse = tmp_t.inverse()
        # print(tmp_inverse)
        # tmp_t2 = torch.eye(4)
        # tmp_t2[:3, :3]  = viewpoint.R_gt.clone()
        # tmp_t2[:3, 3]   = viewpoint.T_gt.clone()
        # print(tmp_t2[:3, 3])
        # tmp_tt = tmp_t2@tmp_inverse
        # tmp_R = tmp_tt[:3,:3]
        # tmp_T = tmp_tt[:3,3]        
        # print(viewpoint.R)
        # print(viewpoint.T)
        # viewpoint.update_RT(tmp_R,tmp_T)  
        # print(viewpoint.R)
        # print(viewpoint.T)
        msg = ["new_map", cur_frame_idx,viewpoint,depth_map]
        # msg = ["new_map", cur_frame_idx,viewpoint,last_kf,depth_map]
        self.backend_queue.put(msg)
        print("[f-n]last kf = %i" % self.last_kf)
        # print("f  : tag = new map")
        self.requested_new_submap = True
        

        
    def request_reset_submap(self, cur_frame_idx, viewpoint,depth_map):
                 
        msg = ["reset", cur_frame_idx,viewpoint,depth_map]
        # msg = ["new_map", cur_frame_idx,viewpoint,last_kf,depth_map]
        self.backend_queue.put(msg)
        print("f  : tag = reset sub map")
        self.requested_new_submap = True  

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        
        # print("f  : tag = init")
        self.requested_init = True

    def sync_backend(self, data):
        
        if data[0] == "init" or data[0] =="new_map" or data[0] =="reset":
            
            self.active_submap = data[1]              
            self.last_kf = self.active_submap.current_window[0]
            
            for kf_id in self.active_submap.current_window:
                kf = self.active_submap.viewpoints[kf_id]           
                self.cameras[kf_id].update_RT(kf.R.clone(), kf.T.clone())      
            
            
            if  data[0]=="init" and self.use_gui:
               
                self.params_gui = gui_utils.ParamsGUI(
                pipe=self.pipeline_params,
                background=clone_obj(self.active_submap.background),
                gaussians=clone_obj(self.active_submap.gaussians),
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
                )
               
                self.gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
                self.gui_process.start()
                time.sleep(5)            
            
        else :
            
            self.active_submap.gaussians = data[1]    
            self.active_submap.occ_aware_visibility  = data[2]           
            keyframes = data[3]
            self.last_kf = self.active_submap.current_window[0]       
        
            for kf_id, kf_R, kf_T in keyframes:
                # if kf_id == self.active_submap.kf_idx[0] :
                #     tmp_t = torch.eye(4)
                #     tmp_t[:3, :3]  = kf_R.clone()
                #     tmp_t[:3, 3] = kf_T.clone()
                #     tmp_inverse = tmp_t.inverse()
                #     self.active_submap.T_CW = tmp_inverse@self.active_submap.original_TCW                
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
        torch.cuda.empty_cache()

    def submap_to_cpu(self, submap):
        # msg = ["reset_mem"]        
        # self.backend_queue.put(msg)
        for keys,views in submap.viewpoints.items():
            self.cameras[keys].viewpoints_to_cpu()
            views.viewpoints_to_cpu()
        
        for idx in submap.current_window :
            submap.occ_aware_visibility[idx] = None        
            torch.cuda.empty_cache()
        submap.gaussians.reset()                

    def eval_(self, tag_):
        eval_ate_(                        
            self.active_submap,
            tag_,
            self.save_dir,
            monocular=self.monocular,
        )  
        # eval_rendering_(
        #     self.cameras,
        #     self.active_submap.gaussians,
        #     self.dataset,        
        #     self.pipeline_params,
        #     self.active_submap.background,
        #     kf_indices=self.active_submap.kf_idx,
        #     tag_=tag_,
        # )
        # eval_ate(
        #     self.submap_list,
        #     self.active_submap,
        #     self.save_dir,
        #     self.active_submap.kf_idx[-1],
        #     final=False,
        #     monocular=self.monocular,
        #     new_submap = True,
        #     tag=tag_
        # )  

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
                   
            #여기부터 진짜 시작
            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    self.submap_list.append(self.active_submap)
                    if self.save_results:
                        self.eval_("final_submap_finish")
                        if self.use_gui:
                            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
                            self.gui_process.join()
                            Log("GUI Stopped and joined the main thread")
                        # eval_ate(
                        #     self.submap_list,
                        #     self.active_submap,
                        #     self.save_dir,
                        #     0,
                        #     final=True,
                        #     monocular=self.monocular,
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
                # self.check_mem("before")
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                
                current_window_dict = {}
                current_window_dict[self.active_submap.current_window[0]] = self.active_submap.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.active_submap.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.active_submap.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )               
                
                #tr2 = time.time()
                # self.check_mem("after")
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue                
               
                last_keyframe_idx = self.active_submap.current_window[0]
                # print("last = %i " %last_keyframe_idx)
                #last_keyframe_idx = self.last_kf
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
                    self.active_submap.occ_aware_visibility,
                )
              
                # print("kf test = %r"%create_kf)
                # print("cur vis = %i, last kf = %i " %(curr_visibility.count_nonzero(),self.occ_aware_visibility[last_keyframe_idx].count_nonzero()))
                if len(self.active_submap.current_window) < self.window_size :
                    union = torch.logical_or(
                        curr_visibility, self.active_submap.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    # print(union)
                    intersection = torch.logical_and(
                        curr_visibility, self.active_submap.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    # print(intersection)
                    point_ratio = intersection / union
                    print("point_ratio = %f , check_time= %i" %(point_ratio,check_time))
                    create_kf = (
                        check_time #and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if not create_kf :
                   create_kf = ( (cur_frame_idx - last_keyframe_idx) >= 15 )
                # print("kf test2 = %r"%create_kf)
                # if self.single_thread:
                #     create_kf = check_time and create_kf
                if create_kf:
                    #print("cur frame is keyframe")
                    #self.kf_indices.append(cur_frame_idx)
                    # d1 = time.time()
                    # print("%r" %(self.active_submap.gaussians.optimizer is None))
                    # d2= time.time()
                    # print("depth map time = "+str(d2-d1))
                    create_new_submap = self.is_new_submap(cur_frame_idx)         
                    #cur_frame_idx, last_keyframe_idx, curr_visibility,
                    #self.occ_aware_visibility,viewpoint,depth_map)
                    if(create_new_submap):
                        
                        # print("-"*60)
                        # self.eval_("before")
                        
                        
                        # # depth_map = self.add_new_keyframe(cur_frame_idx, viewpoint,depth=render_pkg["depth"], init=False)
                        depth_map = self.add_new_keyframe(
                            cur_frame_idx,                                                    
                            depth=render_pkg["depth"],
                            opacity=render_pkg["opacity"],
                            init=False,
                        )    
                        d1= time.time()
                        # depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
                        d2= time.time()
                        self.request_new_submap(cur_frame_idx,viewpoint,depth_map)
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
                            self.active_submap.occ_aware_visibility,
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
                            depth=render_pkg["depth"],
                            opacity=render_pkg["opacity"],
                            init=False,
                        )                           
                                    
                        self.request_keyframe(
                            cur_frame_idx, viewpoint, self.active_submap.current_window, depth_map, self.active_submap.kf_idx)
                    
                        print("kf idx = ",end="")
                        for i  in self.kf_indices :
                            print(" %i"%i,end="")
                        print(" [%i]" %len(self.kf_indices))       
                                              
                else:
                    # print("cur frame is not keyframe!")
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1
                
                if (
                    self.save_results
                    and len(self.kf_indices) >=5
                    and not create_new_submap
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                    ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    # print("len of kf indices = %i "%len(self.kf_indices))
                    self.eval_("before")
                    eval_ate(                        
                        self.submap_list,
                        self.active_submap,
                        self.save_dir,
                        cur_frame_idx,
                        False,
                        monocular=self.monocular,
                    )  
                    
                toc.record()               
                torch.cuda.synchronize()               
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                print("b num of queue = %i " %self.frontend_queue.qsize())
                data = None
                if (self.frontend_queue.qsize() ==1):
                    data = self.frontend_queue.get()    
                while self.frontend_queue.qsize() >=1 :
                    data = self.frontend_queue.get()
                    if data[0]=="sync_backend":
                        continue
                    else :
                        break                
                print("a num of queue = %i, tag = %s " %(self.frontend_queue.qsize(),data[0]))
                
                # data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    print("bf : tag = sync_backend")
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    print("bf  : tag = keyframe")
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    print("bf : tag = init")
                    self.sync_backend(data)
                    self.requested_init = False
                    
                elif data[0] == "new_map":
                    print("bf : tag = new_map")
                    self.sync_backend(data)
                    self.requested_new_submap = False 
                    self.initialized = not self.monocular      
                
                elif data[0] == "refine":
                    print("bf : tag = refine")
                    self.sync_backend(data)                  
                    self.eval_("after")
                    print("-"*60)
                    
                elif data[0] == "reset":
                    print("bf : tag = reset")
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