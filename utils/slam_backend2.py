import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.submap_utils import Submap


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.active_submap=None 
        self.pipeline_params = None
        self.opt_params = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False        
        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.submap_init = False
        self.first_ = True
        self.iteration_count = 0
        self.last_sent = 0
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.submap_id = 0 #각  submap마다의  id
        self.global_pose_list =[]
        self.total_kf_indices = []

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.active_submap.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
    
    # def reset(self):
    #     self.iteration_count = 0
    #     self.initialized = not self.monocular
    #     self.keyframe_optimizers = None

    #     # remove all gaussians
    #     self.active_submap.gaussians.prune_points(self.active_submap.gaussians.unique_kfIDs >= 0)
    #     # remove everything from the queues
    #     while not self.backend_queue.empty():
    #         self.backend_queue.get()

    def initialize_sub_map(self, viewpoint, first_ = True ,last_kf = None):
        # if (first_):
            
        self.active_submap = Submap(self.config,self.device,self.submap_id,first_)
        self.active_submap.initialize_(viewpoint)
        # self.global_pose_list.append(self.active_submap.get_anchor_frame_pose())
            # self.submap_list.append(self.active_submap)
            
        # else :
            
        #     self.active_submap = Submap(self.config,self.device,self.submap_id,first_)
        #     self.active_submap.initialize_(viewpoint)#,last_kf)           
        #     self.global_pose_list.append(self.active_submap.get_anchor_frame_pose())
      
    def initialize_map(self,viewpoint):      
        # ck = True
        # for idx, viewpoint in self.active_submap.viewpoints.items():
        for mapping_iteration in range(self.active_submap.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )  
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()        
            with torch.no_grad():
                self.active_submap.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.active_submap.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.active_submap.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.active_submap.init_gaussian_update == 0 : #and ck:
                    self.active_submap.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.active_submap.init_gaussian_th,
                        self.active_submap.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.active_submap.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.active_submap.gaussians.reset_opacity()

                self.active_submap.gaussians.optimizer.step()
                self.active_submap.gaussians.optimizer.zero_grad(set_to_none=True)
        # ck = False  
        # print("f gaussian num = %i" %len(self.active_submap.gaussians._xyz))
        # print(n_touched.shape)  
        self.active_submap.occ_aware_visibility[viewpoint.uid] = (n_touched > 0).long()                              

        Log("Initialized sub map %i "%self.submap_id)
            # to do - 이전 gaussian 추종 
        self.submap_id+=1
        return render_pkg
        
    def map(self, submap, prune=False, iters=1):
        # print("map!")
        current_window = submap.current_window
        if len(current_window) == 0:
            print("current_window size = 0")
            return

        viewpoint_stack = [self.active_submap.viewpoints[kf_idx] for kf_idx in current_window]

        
        random_viewpoint_stack = self.get_all_viewpoint() # random frame뽑을 때 inactvie submap에서만 or not
        frames_to_optimize = self.config["Training"]["pose_window"]        
        # print("map1!")    
        for _ in range(iters): 
            # print("map iter!")   
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                # print("map inner iter!")  
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render( viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                # print("map inner iter rendering!")
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                # print("map inner iter get loss mapping!")  
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                # print("map inner iter end")  
            # print("current_mapping loss")
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
            # print("random_mapping loss")
            scaling = self.active_submap.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            # print("map2!")    
            #########################
            with torch.no_grad():
                self.active_submap.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.active_submap.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                # print("map3!")    
                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    # print("map4!")    
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.active_submap.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.active_submap.occ_aware_visibility.items():
                            self.active_submap.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.active_submap.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.active_submap.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.active_submap.gaussians.unique_kfIDs >= 0 
                            to_prune = torch.logical_and(
                                self.active_submap.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.active_submap.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.active_submap.occ_aware_visibility[current_idx] = (
                                    self.active_submap.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.active_submap.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.active_submap.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.active_submap.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )
                # print("map5!")    
                update_gaussian = (
                    self.iteration_count % self.active_submap.gaussian_update_every
                    == self.active_submap.gaussian_update_offset
                )
                if update_gaussian:
                    # print("map6!")    
                    self.active_submap.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.active_submap.gaussian_th,
                        self.active_submap.gaussian_extent,
                        self.active_submap.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.active_submap.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.active_submap.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                # print("map7!")    
                self.active_submap.gaussians.optimizer.step()
                self.active_submap.gaussians.optimizer.zero_grad(set_to_none=True)
                self.active_submap.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
                # print("map8!")    
        return gaussian_split    

    # def get_all_viewpoint(self):
    #     total_view = []
    #     for submap_ in self.submap_list :
    #         for idx,kf in submap_.viewpoints.items():              
    #             if idx in self.active_submap.current_window :
    #                 continue
    #             if idx in total_view :
    #                 continue 
    #             total_view.append(kf)
    #     return total_view

    def get_all_viewpoint(self):
        
        total_view = []
        for idx,kf in self.active_submap.viewpoints.items():              
            if idx in self.active_submap.current_window :
                continue
            total_view.append(kf)
        return total_view

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        # for kf_idx in self.active_submap.current_window:
        #     kf = self.active_submap.viewpoints[kf_idx]
        #     keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        
        if tag is None:
            tag = "sync_backend"
        
        # print("tag = "+tag)
        # if(tag =="new_map"):
        #     print('-------backend-------')
           
        #     print("active_submap size = %i , cur_win_size = %i"%(self.active_submap.get_submap_size(),self.active_submap.get_win_size()))
        #     print("active sub map idx [kf_idx] = ",end="")
        #     for i  in self.active_submap.kf_idx :
        #         print(" %i"%i,end="")
        #     print(" ")
        #     print("active sub map idx [viewpoints] = ",end="")
        #     for idx, view in self.active_submap.viewpoints.items() :
        #         print(" %i"%idx,end="")
        #     print(" ")
        #     print("active sub map idx [occ_aware_visibility] = ",end="")
        #     for idx, view in self.active_submap.occ_aware_visibility.items() :
        #         print(" %i"%idx,end="")
        #     print(" ")            
        #     print("current_window idx = ",end="")
        #     for i  in self.active_submap.current_window :
        #         print(" %i"%i,end="")
        #     print(" ")    
        msg = [tag, clone_obj(self.active_submap)]
        # msg = [tag, clone_obj(self.active_submap.gaussians), self.active_submap.occ_aware_visibility, keyframes, clone_obj(self.active_submap)]        self.frontend_queue.put(msg)
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                # print("112")
                # print("pause = %r"%self.pause)
                if self.pause:
                    # print("pause!!")
                    time.sleep(0.01)
                    continue
                # print("winsize = %s"%self.active_submap.get_win_size())
                # if (self.active_submap.get_win_size()) == 0:
                #     time.sleep(0.01)
                #     continue
                # print(self.active_submap.init_)
                if (not self.submap_init) :
                    time.sleep(0.01)
                    continue

                # if self.single_thread:
                #     time.sleep(0.01)
                #     continue
                
                self.map(self.active_submap)
                if self.last_sent >= 10:
                    self.map(self.active_submap, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                
                data = self.backend_queue.get()
                # print(data[0])
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "new_map":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    self.iteration_count = 0
                    self.last_sent = 0
                    self.initialized = not self.monocular
                    # self.submap_init = False
                    # self.submap_list.append(self.active_submap)
                    self.initialize_sub_map(viewpoint,False)#,last_kf)                                   
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    ) 
                    self.initialize_map(viewpoint)                                   
                    self.push_to_frontend("new_map")
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    # self.reset()
                    self.initialize_sub_map(viewpoint,True) 
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )   
                    self.initialize_map(viewpoint)#cur_frame_idx,viewpoint)                                    
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    # print("keyframe1")
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    kf_idx = data[5]         
                    self.active_submap.viewpoints[cur_frame_idx] = viewpoint
                    self.active_submap.current_window = current_window
                    self.active_submap.kf_idx = kf_idx
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    self.submap_init = True
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = 10
                    if not self.initialized:
                        if (
                            len(self.active_submap.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.active_submap.mapping_itr_num
                    for cam_idx in range(len(self.active_submap.current_window)):
                        if self.active_submap.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.active_submap.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    # print("keyframe2")
                    self.map(self.active_submap, iters=iter_per_kf)
                    # print("keyframe3")
                    self.map(self.active_submap, prune=True)
                    # print("keyframe4")
                    self.push_to_frontend("keyframe")
                    # print("keyframe5")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        print("real end!!1")
        return
