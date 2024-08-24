import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
import random
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend2 import BackEnd
from utils.slam_frontend2_baqueue import FrontEnd
from utils.slam_BA import BA
from utils.multiprocessing_utils import clone_obj

class SLAM:
    def __init__(self, config, save_dir=None):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)           
        start.record()
     
        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )    
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0     
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
      
        # bg_color = [0, 0, 0]
        # self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.submap_list = []
        # manager= mp.Manager()
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        before_BA_queue = mp.Queue()        
        
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)
        self.BA = BA(self.config)
        self.BA.before_ba_queue = before_BA_queue
        self.BA.pipeline_params = self.pipeline_params
        self.BA.dataset = self.dataset

        self.frontend.dataset = clone_obj(self.dataset)
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.opt_params = self.opt_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.before_ba_queue = before_BA_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()
    
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode        

        backend_process = mp.Process(target=self.backend.run)    
        frontend_process = mp.Process(target=self.frontend.run)    

        backend_process.start()       
        frontend_process.start()    
        self.BA.run()         

        print("BA run break!")
        backend_queue.put(["pause"])
        
        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.BA.cameras)
        FPS = 0.1
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
        total_submap_num = len(self.BA.submap_list)
        if total_submap_num ==0 :
            total_submap_num=1
        print("submap num = %i"%total_submap_num)
        print("final kf num = %i" %len(self.BA.kf_indices))
        if self.eval_rendering:
            ATE = eval_ate(
                    self.BA.submap_list,
                    self.BA.last_active_submap,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )
            # total_submap_num = len(self.frontend.submap_list)
            total_psnr = 0
            total_ssim = 0
            total_lpips = 0
            total_frame_num = 0
            for submap_ in self.BA.submap_list:              
        
                rendering_result = eval_rendering(
                    self.BA.cameras,                 
                    submap_.gaussians,
                    self.BA.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    submap_.background,
                    kf_indices=submap_.kf_idx ,
                    iteration="before_opt",
                )
                total_psnr+=rendering_result["mean_psnr"]*rendering_result["total frame num"]
                total_ssim +=rendering_result["mean_ssim"]*rendering_result["total frame num"]
                total_lpips +=rendering_result["mean_lpips"]*rendering_result["total frame num"]  
                total_frame_num +=rendering_result["total frame num"]
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                total_psnr/total_frame_num,
                total_ssim/total_frame_num,
                total_lpips/total_frame_num,
                ATE,
                FPS,
            )
            while not frontend_queue.empty():
                frontend_queue.get()       
            print("before_psnr = %f" %float(total_psnr/total_frame_num))
   
        backend_queue.put(["stop"])
        backend_process.join()
        frontend_process.join()
        Log("Backend stopped and joined the main thread")


    def run(self):
        pass
   

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
