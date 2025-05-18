import torch 
import torch.nn as nn
from torchvision.utils import make_grid
from glow.glow import Glow
import numpy as np
import skimage.io as sio
import os
import json
import argparse


def sampleGlow(args):
    save_path   = f"./trained_models/{args.dataset}/glow_{args.size}_{args.job_id}/"
    
    # setting up configs as json
    config_path = save_path+"/configs.json"
    
    _ = torch.inverse(torch.ones((1, 1), device=args.device))  # init linear algebra to avoid lazy wrapper errors
    
    # loading pre-trained model to resume training
    model_path = save_path + "glowmodel.pt"
    if os.path.exists(model_path):
        print("loading previous model and saved configs to resume training ...")
        with open(config_path, 'r') as f:
            configs = json.load(f)

        glow = Glow((1,configs["size"],configs["size"]), 
                    K=configs["K"], L=configs["L"], coupling=configs["coupling"],
                    device=args.device, 
                    n_bits_x=configs["n_bits_x"], 
                    nn_init_last_zeros=configs["last_zeros"])
        if args.device == 'cpu':
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(model_path)
            
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # 去掉前缀
            new_state_dict[new_key] = v
                
        glow.load_state_dict(new_state_dict)

    
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please check the path.")

    
    for i in range(args.epochs):
        print("Epoch %d/%d"%(i+1,args.epochs))
        with torch.no_grad():
            glow(torch.randn(1, 1, 128, 128).to(args.device))
            z_sample, z_sample_t = glow.generate_z(n=10,mu=0,std=0.7,to_torch=True)
            x_gen = glow(z_sample_t, reverse=True)
            x_gen = glow.postprocess(x_gen)
            x_gen = make_grid(x_gen,nrow=int(np.sqrt(len(x_gen))))
            x_gen = x_gen.data.cpu().numpy()
            x_gen = x_gen.transpose([1,2,0])
            if x_gen.shape[-1] == 1:
                x_gen = x_gen[...,0]
            if not os.path.exists(save_path+"/samples_training"):
                os.makedirs(save_path+"/samples_training")
            x_gen = (np.clip(x_gen, 0, 1) * 255).astype("uint8")
            sio.imsave(save_path+"/samples_training/%0.6d.jpg"%i, x_gen)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train glow network')
    parser.add_argument('-dataset',type=str,help='the dataset to train the model on', default='celeba')
    parser.add_argument('-size',type=int,help='images will be resized to this dimension',default=64)
    parser.add_argument('-lr',type=float,help='learning rate for training',default=1e-4)
    parser.add_argument('-epochs',type=int,help='epochs to train for',default=10)
    parser.add_argument('-device',type=str,help='whether to use',default="cuda")  
    parser.add_argument('-job_id', type=str, help='job id to save the model', default="0")  
    args = parser.parse_args()
    # Try to initialize CUDA, fallback to CPU if fails
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        # force trigger device count to check CUDA runtime
        torch.cuda.device_count()
        print("CUDA initialized successfully.")
        print(f"Using device: {args.device}, GPU count: {torch.cuda.device_count()}")
            
    sampleGlow(args)
    
