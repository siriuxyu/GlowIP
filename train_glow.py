import torch 
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from glow.glow import Glow
import numpy as np
import skimage.io as sio
import matplotlib.pyplot as plt
import os
import json
import argparse
import re
from collections import defaultdict
from data.npzdata import NPZDataset


def trainGlow(args):
    multiGPU = False
    if torch.cuda.device_count() > 1:
        multiGPU = True
        
    save_path   = f"./trained_models/{args.dataset}/glow_{args.size}_{args.job_id}/"
    training_folder = f"./data/{args.dataset}_preprocessed/train/"
    npz_file     = f"./data/test_data/{args.dataset}/{args.dataset}_train.npz"
    
    # setting up configs as json
    config_path = save_path+"/configs.json"
    configs     = {"K":args.K,
                   "L":args.L,
                   "coupling":args.coupling,
                   "last_zeros":args.last_zeros,
                   "batchsize":args.batchsize,
                   "size":args.size,
                   "lr": args.lr,
                   "n_bits_x":args.n_bits_x,
                   "warmup_iter":args.warmup_iter}

    if(args.squeeze_contig):
        configs["squeeze_contig"] = True
    if(args.coupling_bias > 0):
        configs["coupling_bias"] = args.coupling_bias
    if not os.path.exists(save_path):
        print("creating directory to save model weights")
        os.makedirs(save_path)
    
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
        
        glow.load_state_dict(torch.load(model_path))
        if multiGPU:
            print(f"Using {torch.cuda.device_count()} GPUs")
            glow = torch.nn.DataParallel(glow)

        print("pre-trained model and configs loaded successfully")
        glow.set_actnorm_init()
        print("actnorm initialization flag set to True to avoid data dependant re-initialization")
        glow.train()
    
    else:
        # creating and initializing glow model
        print("creating and initializing model for training")
        glow = Glow((1,args.size,args.size),
                    K=args.K,L=args.L,coupling=args.coupling,
                    device=args.device,
                    n_bits_x=args.n_bits_x, 
                    nn_init_last_zeros=args.last_zeros)
        glow.train()
        
        # Multi-GPU support
        if multiGPU:
            print(f"Using {torch.cuda.device_count()} GPUs")
            glow = torch.nn.DataParallel(glow)
        
        print("saving configs as json file")
        with open(config_path, 'w') as f:
            json.dump(configs, f, sort_keys=True, indent=4, ensure_ascii=False)
            
    
    # setting up dataloader
    print("setting up dataloader for the training data")
    if args.dataset == "celeba":
        trans      = transforms.Compose([transforms.Resize(args.size),
                                        transforms.CenterCrop((args.size, args.size)),
                                        transforms.ToTensor()])
        dataset    = datasets.ImageFolder(training_folder, transform=trans)
    if args.dataset in ["BraTS", "LDCT", "LIDC_320", "LIDC_512"]:
        dataset    = NPZDataset(npz_file, size=args.size)
        dataset.sample_images(args.n_data)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                                                drop_last=True, shuffle=True)
    
    
    
    # setting up optimizer and learning rate scheduler
    lr_adjusted = args.lr * (64/args.size)**2 * (args.batchsize/4)
    opt          = torch.optim.Adam(glow.parameters(), lr=lr_adjusted)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",
                                                              factor=0.5,
                                                              patience=1000,
                                                              min_lr=1e-8)
    
    # extract module from DataParallel if used
    core_glow = glow.module if isinstance(glow, torch.nn.DataParallel) else glow
    
    # starting training code here
    print("+-"*10,"starting training","-+"*10)
    global_step = 0
    global_loss = []
    warmup_completed = False
    for i in range(args.epochs):
        for j, data in enumerate(dataloader):
            opt.zero_grad()
            core_glow.zero_grad()
            # loading batch
            x = data.to(device=args.device)*255
            # pre-processing data
            x = core_glow.preprocess(x)
            # computing loss: "nll"
            nll,logdet,logpz,z_mu,z_std = glow(x)
            # skipping first batch due to data dependant initialization (if not initialized)
            if global_step == 0:
                global_step += 1
                continue
            # backpropogating loss and gradient clipping
            nll.mean().backward()
            torch.nn.utils.clip_grad_value_(core_glow.parameters(), 5)
            grad_norm = torch.nn.utils.clip_grad_norm_(core_glow.parameters(), 100)
            # linearly increase learning rate till warmup_iter upto args.lr
            if global_step <= args.warmup_iter:
                warmup_lr = args.lr / args.warmup_iter * global_step
                for params in opt.param_groups:
                    params["lr"] = warmup_lr
            # taking optimizer step                        
            opt.step()
            # learning rate scheduling after warm up iterations
            if global_step > args.warmup_iter:
                lr_scheduler.step(nll.mean())
                if not warmup_completed:
                    if args.warmup_iter == 0:
                        print("no model warming...")
                    else:
                        print("\nwarm up completed")
                warmup_completed = True
            # printing training metrics 
            print("\repoch=%0.2d..nll=%0.2f..logdet=%0.2f..logpz=%0.2f..mu=%0.2f..std=%0.2f..gradnorm=%0.2f"
                  %(i,nll.mean().item(), logdet.mean().item(), logpz.mean().item(),
                    z_mu.mean().item(), z_std.mean().item(), grad_norm),end="\r")
            # saving generated samples during training
            try:
                if global_step % args.sample_freq == 0:
                    plt.plot(global_loss)
                    plt.xlabel("iterations",size=15)
                    plt.ylabel("nll",size=15)
                    plt.savefig(save_path+"/nll_training_curve.jpg")
                    plt.close()
                    print("\n saving generated samples at global step = %d"%global_step)
                    with torch.no_grad():
                        with torch.no_grad():
                            core_glow(torch.randn(1, 1, 128, 128).to(args.device))
                        z_sample, z_sample_t = core_glow.generate_z(n=10,mu=0,std=0.7,to_torch=True)
                        x_gen = glow(z_sample_t, reverse=True)
                        x_gen = core_glow.postprocess(x_gen)
                        x_gen = make_grid(x_gen,nrow=int(np.sqrt(len(x_gen))))
                        x_gen = x_gen.data.cpu().numpy()
                        x_gen = x_gen.transpose([1,2,0])
                        if x_gen.shape[-1] == 1:
                            x_gen = x_gen[...,0]
                        if not os.path.exists(save_path+"/samples_training"):
                            os.makedirs(save_path+"/samples_training")
                        x_gen = (np.clip(x_gen, 0, 1) * 255).astype("uint8")
                        sio.imsave(save_path+"/samples_training/%0.6d.jpg"%global_step, x_gen)
            except Exception as e:
                print(e)
                print("\n failed to sample from glow at global step = %d"%global_step)
            global_step = global_step + 1
            global_loss.append(nll.mean().item())
            if global_step % args.save_freq == 0:
                torch.save(core_glow.state_dict(), model_path)
                
    # saving model weights
    torch.save(glow.state_dict(), model_path)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train glow network')
    parser.add_argument('-dataset',type=str,help='the dataset to train the model on', default='celeba')
    parser.add_argument('-K',type=int,help='no. of steps of flow',default=32)
    parser.add_argument('-L',type=int,help='no. of time squeezing is performed',default=6)
    parser.add_argument('-coupling',type=str,help='type of coupling layer to use',default='affine')
    parser.add_argument('-last_zeros',type=bool,help='whether to initialize last layer ot NN with zeros',default=True)
    parser.add_argument('-n_data',type=int,help='no. of training images to use',default=800)
    parser.add_argument('-batchsize',type=int,help='batch size for training',default=3)
    parser.add_argument('-size',type=int,help='images will be resized to this dimension',default=64)
    parser.add_argument('-lr',type=float,help='learning rate for training',default=1e-5)
    parser.add_argument('-n_bits_x',type=int,help='requantization of training images',default=5)
    parser.add_argument('-epochs',type=int,help='epochs to train for',default=1000)
    parser.add_argument('-warmup_iter',type=int,help='no. of warmup iterations',default=10000)
    parser.add_argument('-sample_freq',type=int,help='sample after every save_freq',default=50)
    parser.add_argument('-save_freq',type=int,help='save after every save_freq',default=1000)
    parser.add_argument('-coupling_bias', type=float,help='additive bias to the scale parameter of each affine coupling layer to prevent division by eps', default=0)
    parser.add_argument('-squeeze_contig', action="store_true", help="whether to select contiguous components of activations in each squeeze layer")
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
            
    trainGlow(args)
    
