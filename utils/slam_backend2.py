import random
import time
import gc
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim , l2_loss , l1_loss_log_pow , l2_loss_log
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_tracking , depth_reg
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
    def reset_submap(self,viewpoint,cur_idx):
        
        self.active_submap.viewpoints = {}
        self.active_submap.viewpoints[cur_idx] = viewpoint
        self.active_submap.kf_idx = []
        self.active_submap.kf_idx.append(cur_idx)
        self.active_submap.current_window = []
        self.active_submap.current_window.append(cur_idx)  
        self.active_submap.occ_aware_visibility={}   
        
    def reset(self):
   
        # self.keyframe_optimizers = None

        # remove all gaussians
        self.active_submap.reset()
        self.active_submap = None   
   
    def color_refinement(self , iteration_total):
        Log("Starting color refinement")

        print(self.active_submap.gaussians._xyz[0])
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.active_submap.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            # print(viewpoint_cam_idx)
            
            viewpoint_cam = self.active_submap.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.active_submap.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.active_submap.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.active_submap.gaussians.optimizer.step()
                self.active_submap.gaussians.optimizer.zero_grad(set_to_none=True)
                self.active_submap.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")
        print(self.active_submap.gaussians._xyz[0]) 
    

    def initialize_sub_map(self, viewpoint, first_ = True): # ,last_kf = None):
        if (first_):    
           
            self.active_submap = Submap(self.config,self.device,self.submap_id,first_)
            self.active_submap.initialize_(viewpoint)
            # self.global_pose_list.append(self.active_submap.get_anchor_frame_pose())
            # self.submap_list.append(self.active_submap)
            
        else :
                    
            last_pose = self.active_submap.get_anchor_frame_pose() 
            self.submap_to_cpu(self.active_submap)
            # r1 = time.time()
            # # self.reset()
            # r2= time.time()
            # print("reset time = %s"%str(r2-r1))
            self.active_submap = Submap(self.config,self.device,self.submap_id,first_)
            self.active_submap.initialize_(viewpoint,last_pose)#,last_kf)           
            #self.global_pose_list.append(self.active_submap.get_anchor_frame_pose())
      
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
                if mapping_iteration % self.active_submap.init_gaussian_update == 0 :#and ck:
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
    
    def set_pose_optimizer(self,selected_idx) :
        opt_params = []
        for id in range(len(selected_idx)):
            viewpoint = self.active_submap.viewpoints[self.active_submap.kf_idx[selected_idx[id]]]
            if id>=( len(selected_idx)-3 ) :
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
        pose_optimizers = torch.optim.Adam(opt_params)
        return pose_optimizers
    
    def pose_refine(self, iters=500, frames_to_optimize_ = 10):
        # print("map! last_sent =%i "%self.last_sent )
      
        viewpoint_stack = [self.active_submap.viewpoints[kf_id] for kf_id in self.active_submap.kf_idx]  
        kf_num  = len(viewpoint_stack) 
        for s in range(kf_num//frames_to_optimize_ +1):
            if(s == kf_num//frames_to_optimize_):
                if kf_num%frames_to_optimize_==0 :
                    continue
                else :
                    selected_idx = torch.arange(s*frames_to_optimize_, kf_num)                       
            else :
                selected_idx = torch.arange(s*frames_to_optimize_,(s+1)*frames_to_optimize_)       
            random_idx = []
            for idx in range(kf_num):
                if idx in selected_idx:
                    continue
                random_idx.append(idx)
            optimizer_ = self.set_pose_optimizer(selected_idx)  
            for _ in range(iters):          
                loss_mapping = 0             
                t1=time.time()         
                
                for cam_idx in selected_idx:
                    viewpoint = viewpoint_stack[cam_idx]                  
                    render_pkg = render( viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background )
                    (
                        image,               
                        depth,
                        opacity
                    
                    ) = (
                        render_pkg["render"],                
                        render_pkg["depth"],
                        render_pkg["opacity"]              
                    )          
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity
                    )   
                    
                for cam_idx in torch.randperm(len(random_idx))[:2]:
                    viewpoint = viewpoint_stack[random_idx[cam_idx]]                  
                    render_pkg = render( viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background )
                    (
                        image,               
                        depth,
                        opacity
                    
                    ) = (
                        render_pkg["render"],                
                        render_pkg["depth"],
                        render_pkg["opacity"]              
                    )          
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity
                    )   
                t2=time.time()        
                scaling = self.active_submap.gaussians.get_scaling
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                loss_mapping += 10 * isotropic_loss.mean()
                loss_mapping.backward()           
                with torch.no_grad():
                    optimizer_.step()
                    optimizer_.zero_grad(set_to_none=True)
                    t14= time.time()
                    # Pose update
                    for idx  in selected_idx[-3:] :                 
                        update_pose(viewpoint_stack[idx])        
        return 0
    
    def pose_refine_track(self ,refine_iter):
       
        for idx in self.active_submap.kf_idx:
            viewpoint = self.active_submap.viewpoints[idx]
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
            # print("-"*50)
            for tracking_itr in range(refine_iter):  #self.tracking_itr_num):
                
                render_pkg = render(
                    viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )               
                
                # gt_image = viewpoint.original_image.cuda()
                # # print(gt_image.shape)
                # Ll1 = l1_loss(image, gt_image)
                # loss1 = (1.0 - self.opt_params.lambda_dssim) * (
                #     Ll1
                # ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                # loss.backward()
                
                # loss_t = get_loss_tracking(self.config, image, depth, opacity, viewpoint )
                     
                gt_depth = torch.Tensor(viewpoint.depth).unsqueeze(0).cuda()
                gt_depth[torch.where(gt_depth<=1)] =1
                depth[torch.where(depth<=1)] =1
                d1 = l1_loss_log_pow(depth,gt_depth)
                d2 = l2_loss_log(depth, gt_depth)
                d_ = torch.sqrt(d2-0.85*d1)
                # loss2 =  0.6*d_ + 0.2* depth_reg(depth,gt_depth)+(1.0 - ssim(depth, gt_depth))
                # # loss2 = (1.0 - self.opt_params.lambda_dssim) * (
                # #     Ll2
                # # ) + self.opt_params.lambda_dssim * (1.0 - ssim(depth, gt_depth))
                       
                                
                
        
                loss = d_#loss1 #loss2+loss3
                # print(loss)
                loss.backward()
                # loss_tracking = get_loss_mapping(#tracking(
                #     self.config, image, depth, viewpoint,opacity
                # )            
                # loss_tracking.backward()

                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad()
                    converged = update_pose(viewpoint)
        
                if converged:       
                    # print("[converge]trcaking_itr =  %i" %tracking_itr)   
                    break          
        
        return 0
    
    def pose_refine_t(self ,refine_iter,viewpoint,use_prev = False):
        if use_prev:
            prev_kf = self.active_submap.get_last_frame()
            viewpoint.update_RT(prev_kf.R.clone(),prev_kf.T.clone())
       
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
         
         
    def map(self, submap, prune=False, iters=1):
        # print("map! last_sent =%i "%self.last_sent )
        
        current_window = submap.current_window
        if len(current_window) == 0:
            print("current_window size = 0")
            return
        # print("len active key = %i " %len(self.active_submap.viewpoints))
        # print("curent window size = %i "%len(current_window))
        viewpoint_stack = [self.active_submap.viewpoints[kf_idx] for kf_idx in current_window]

        t1 =time.time()
        random_viewpoint_stack = self.get_all_viewpoint() # random frame뽑을 때 inactvie submap에서만 or not
        # print("ss %i " %len(random_viewpoint_stack))
        t2 =time.time()
        frames_to_optimize = self.config["Training"]["pose_window"]        
        # print("map1!")   
        cnt =0 
        for _ in range(iters): 
            # print("map iter!")   
            cnt+=1
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
            t3= time.time()
            for cam_idx in range(len(current_window)):
                # print("map inner iter!") 
                # cam_idx = len(current_window)-1-cam_idx 
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
            
            t4= time.time()
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
            t5=time.time()        
            scaling = self.active_submap.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            t6= time.time()
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
                t7= time.time()
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
                    t8= time.time()
                    
                    # print("--------------map prune-------------")  
                    # print("initialized = %r" %self.initialized)              
                    # print("get_randome_view : " +str(t2-t1))
                    # print("current_opt (1 iter) : " +str(t4-t3))
                    # print("random_opt (1 iter) : " +str(t5-t4))
                    # print("loss backward : " +str(t6-t5))
                    # print("occ vis assign : " +str(t7-t6))
                    # print("prune : " +str(t8-t7))
                    # print("total : "+ str(t8-t1))
                    # print("-"*30)  
                    
                    return False
                
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.active_submap.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.active_submap.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.active_submap.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )
                t9= time.time()
                update_gaussian = (
                    self.iteration_count % self.active_submap.gaussian_update_every
                    == self.active_submap.gaussian_update_offset
                )
                t10= time.time()
                
                if update_gaussian:
                    # print("map6!")    
                    self.active_submap.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.active_submap.gaussian_th,
                        self.active_submap.gaussian_extent,
                        self.active_submap.size_threshold,
                    )
                    gaussian_split = True
                t11= time.time()
                ## Opacity reset
                if (self.iteration_count % self.active_submap.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.active_submap.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                # print("map7!")    
                t12= time.time()
                self.active_submap.gaussians.optimizer.step()
                self.active_submap.gaussians.optimizer.zero_grad(set_to_none=True)
                self.active_submap.gaussians.update_learning_rate(self.iteration_count)
                t13= time.time()
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                t14= time.time()
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    # cam_idx = -1*cam_idx -1
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == self.active_submap.kf_idx[0]:
                        continue
                    update_pose(viewpoint)
                t15= time.time()
   
        return gaussian_split    

    def get_all_viewpoint(self):
        
        total_view = []
        for idx,kf in self.active_submap.viewpoints.items():              
            if idx in self.active_submap.current_window :
                continue
            total_view.append(kf)
        return total_view

    def submap_to_cpu(self, submap):
        for views in submap.viewpoints.values():
            views.viewpoints_to_cpu()
        
        for idx in submap.current_window :
            submap.occ_aware_visibility[idx] = None        
            torch.cuda.empty_cache()
        submap.gaussians.reset() 
    
    def check_mem(self,tag):
        print( tag+ f" Reserved Memory : {torch.cuda.memory_reserved()/1024.0/1024.0:.2f}MB" )
        print( tag+ f" Allocated Memory : {torch.cuda.memory_allocated()/1024.0/1024.0:.2f}MB")
    
    def push_to_frontend2(self, tag=None):
        self.last_sent = 0       
        if tag is None:
            tag = "sync_backend"          
        msg = [tag, clone_obj(self.active_submap)]
        self.frontend_queue.put(msg)
    
    def push_to_frontend(self, tag=None):
        self.last_sent = 0       
        if tag is None:
            tag = "sync_backend"  
        keyframes = []
        for kf_idx in self.active_submap.current_window:
            kf = self.active_submap.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
      
        last_ = self.active_submap.current_window[0]      
        if(tag=="init" or tag=="new_map" or tag=="reset"):
            # print("bb : tag = "+tag)  
            msg = [tag, clone_obj(self.active_submap)]
            self.frontend_queue.put(msg)

        else :
            # msg = [tag, self.active_submap.gaussians, self.active_submap.occ_aware_visibility,keyframes]
            print("[keyframe # %i] bb : tag = %s " %(last_ ,tag))  
            msg = [tag, clone_obj(self.active_submap.gaussians), self.active_submap.occ_aware_visibility,keyframes]
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
                # print(self.active_submap.init_)
                # print("m0")
                if (not self.submap_init) :
                    time.sleep(0.01)
                    continue
                # if self.single_thread:
                #     time.sleep(0.01)
                #     continue               
     
                # m1 = time.time()
                self.check_mem("map_kf back")
                self.map(self.active_submap)
                self.check_mem("map1_kf back")
                # m2 = time.time()
                
                # print("map time = "+str(m2-m1))
                # print("map 0")
                if self.last_sent >= 10:
                    # pm1 = time.time()
                    self.map(self.active_submap, prune=True, iters=10)
                    self.check_mem("map2_kf back")
                    # print("map 1")
                    self.push_to_frontend()
                    # pm2 = time.time()
                    torch.cuda.synchronize()
                    # print("pmap time = "+str(pm2-pm1))
                    # print("")
            else:
                print("num of backend queue = %i " %self.backend_queue.qsize())
                data = self.backend_queue.get()
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
                    self.submap_init = False
                    # self.submap_list.append(self.active_submap)                    
                    n1 =time.time()
                    # c1 = time.time()
                    # self.color_refinement(15000)
                    # c2 = time.time()
                    # print("")
                    # print("clor_refine time = "+str(c2-c1))
                    # print("")    
                    # t1 = time.time()
                    # self.pose_refine(iters = 30,frames_to_optimize_=8)
                    # t2 = time.time()
                    # print("")
                    # print("pose_refine time = "+str(t2-t1))
                    # print("")
                    
                    # t1 = time.time()
                    # self.pose_refine_track(150)
                    # t2 = time.time()
                    # print("")
                    # print("pose_refine_track time = "+str(t2-t1))
                    # print("")
                                    
                    # t1 = time.time()
                    # self.pose_refine_t(150, viewpoint,False)
                    # t2 = time.time()
                    # print("")
                    # print("pose_refine_t time = "+str(t2-t1))
                    # print("")
                    # self.push_to_frontend("refine")
                    print("----")
                    print(viewpoint.R)
                    print(viewpoint.T)
                    self.initialize_sub_map(viewpoint,False)  
                    
                    print(viewpoint.R)
                    print(viewpoint.T)
                    print("////")
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )                                  
                    self.initialize_map(viewpoint)                                                       
                    self.push_to_frontend("new_map")
                    n2 =time.time()
                    print("")
                    print("new map time = "+str(n2-n1))
                    print("")
                elif data[0] == "reset":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    self.iteration_count = 0
                    self.last_sent = 0
                    self.initialized = False #not self.monocular
                    self.submap_init = False
                    # self.submap_list.append(self.active_submap)                    
                    n1 =time.time()
                    self.reset_submap(viewpoint,cur_frame_idx)
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )                                  
                    self.initialize_map(viewpoint)                                                       
                    self.push_to_frontend("reset")
                    n2 =time.time()
                    print("")
                    print("reset, and initialize new map time = "+str(n2-n1))
                    print("")
                # elif data[0] == "color_refinement":
                #     self.color_refinement()
                #     self.push_to_frontend()
                elif data[0] == "init":
                    i1= time.time()
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    self.submap_init = False
                    Log("Resetting the system")
                    # self.reset()                    
                 
                    self.initialize_sub_map(viewpoint,True) 
         
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )  
           
                    self.initialize_map(viewpoint)#cur_frame_idx,viewpoint)   
                                                      
                    self.push_to_frontend("init")
                    i2= time.time()
                    print("")
                    print("init time = "+str(i2-i1))
                    print("")

                elif data[0] == "keyframe":
                    # print("keyframe1")
                    k1 = time.time()
                    # print("%r" %self.initialized)
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    kf_idx = data[5]         
                    self.check_mem("before0_kf back")
                    self.active_submap.viewpoints[cur_frame_idx] = viewpoint
                    self.active_submap.current_window = current_window
                    self.active_submap.kf_idx = kf_idx
                    self.check_mem("before_kf back")
                    self.add_next_kf(cur_frame_idx, self.active_submap.viewpoints[cur_frame_idx], depth_map=depth_map)
                    self.check_mem("after_kf back")
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
                        
                        # cam_idx = len(self.active_submap.current_window)-1-cam_idx
                        
                        if self.active_submap.current_window[cam_idx] == 0:
                            continue
                       
                        viewpoint = self.active_submap.viewpoints[current_window[cam_idx]]
                        
                        if (cam_idx) < frames_to_optimize:
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
                    self.check_mem("after_kf back2")
                    km1= time.time()
                    self.map(self.active_submap, iters=iter_per_kf)
                    km2= time.time()                  
                    self.map(self.active_submap, prune=True)
                    km3= time.time()
                    self.check_mem("after_kf back3")
                    self.push_to_frontend("keyframe")
                    k2 = time.time()   
                    torch.cuda.empty_cache()     
                    # print("")   
                    # print("km1 time = "+str(km2-km1))
                    # print("km2 time = "+str(km3-km2))
                    # print("push time = "+str(k2-km3))
                    # print("keyframe time = "+str(k2-k1))
                    # print("")
                    # print("keyframe5")
                else:                    
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        print("real end!!1")
        return
    
    
    
        #         print("--------------map iter : %i ---------------"%cnt)                
        #         print("get_randome_view : " +str(t2-t1))
        #         print("current_opt (1 iter) : " +str(t4-t3))
        #         print("random_opt (1 iter) : " +str(t5-t4))
        #         print("loss backward : " +str(t6-t5))
        #         print("occ vis assign : " +str(t7-t6))
        #         print("add densification stat : " +str(t9-t7))
        #         print("update_gaussian decision : " +str(t10-t9))
        #         print("densify and prune : " +str(t11-t10))
        #         print("resetting the opactiy : " +str(t12-t11))
        #         print("gaussian optimizer step : " +str(t13-t12))           
        #         print("keyframe optimizer step : " +str(t14-t13))
        #         print("update pose : " +str(t15-t14)) 
        #         print("1 iter = " + str(t15-t3))          
        #         print("-"*30)  
                
        # print("total = "+ str(t15-t1))
        # print("-----------------------------/")         
                # print("map8!")  