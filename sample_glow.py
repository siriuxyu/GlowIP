import torch 
import torch.nn as nn
from torchvision.utils import make_grid
from glow.glow import Glow
import numpy as np
import skimage.io as sio
import os
import json
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



class PseudoKSpace(torch.nn.Module):
    """
        forward(img, mode="to_k")    # image -> 2-chan k-space
        forward(k2c, mode="to_img")  # 2-chan k-space -> image
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:

        if not reverse:
            return self.to_kspace(x)
        
        elif reverse:
            return self.to_image(x)

    # ---------- image -> 2-channel k-space ----------
    def to_kspace(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, 1, H, W) or (B, H, W) or (H, W)
        return: (B, 2, H, W) or (2, H, W)
        """
        single = img.ndim == 2           # 记录是否无 batch 无 channel
        if single:
            img = img[None, None]        # -> (1,1,H,W)
        elif img.ndim == 3:              # (B,H,W)
            img = img[:, None]           # -> (B,1,H,W)

        # 2D FFT
        k = torch.fft.fftshift(torch.fft.fft2(img.float(), norm="ortho"))

        # 分离实虚并做 0-1 归一化（逐样本）
        real, imag = k.real, k.imag
        r_min, r_max = real.amin(dim=(-2, -1), keepdim=True), real.amax(dim=(-2, -1), keepdim=True)
        i_min, i_max = imag.amin(dim=(-2, -1), keepdim=True), imag.amax(dim=(-2, -1), keepdim=True)
        print(f"r_min: {r_min.shape}, r_max: {r_max.shape}, i_min: {i_min.shape}, i_max: {i_max.shape}")

        real_n = (real - r_min) / (r_max - r_min).clamp_min(self.eps)
        imag_n = (imag - i_min) / (i_max - i_min).clamp_min(self.eps)
        print(f"real_n: {real_n.shape}, imag_n: {imag_n.shape}")

        k2c = torch.cat([real_n, imag_n], dim=1)   # (B,2,H,W)

        # 把 min/max 存到 buffer 里，方便反归一化（也可 return 字典更灵活）
        self.r_min = r_min.squeeze(1)
        self.r_max = r_max.squeeze(1)
        self.i_min = i_min.squeeze(1)
        self.i_max = i_max.squeeze(1)

        if single:
            k2c = k2c[0]  # -> (2,H,W)
        return k2c

    # ---------- 2-channel k-space -> image ----------
    def to_image(self, k2c: torch.Tensor) -> torch.Tensor:
        """
        k2c: (B,2,H,W) or (2,H,W)
        return: (B,1,H,W) or (H,W)
        """
        single = k2c.ndim == 3           # (2,H,W)
        if single:
            k2c = k2c[None]              # -> (1,2,H,W)

        real_n, imag_n = k2c[:, 0], k2c[:, 1]
        

        # 反归一化
        real = real_n * (self.r_max - self.r_min) + self.r_min
        imag = imag_n * (self.i_max - self.i_min) + self.i_min
        print(f"real: {real.shape}, imag: {imag.shape}")

        # 复数重组 & IFFT
        k_complex = torch.complex(real, imag)
        print(f"k_complex: {k_complex.shape}")
        img = torch.fft.ifft2(torch.fft.ifftshift(k_complex), norm="ortho").real  # (B,H,W)

        img = img[:, None]  # -> (B,1,H,W)

        if single:
            img = img[0, 0]  # -> (H,W)
        return img
 


def sampleGlow(args):
    if args.dataset == "BraTS_png":
        channels = 1
    elif args.dataset == "celeba":
        channels = 3
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
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

        glow = Glow((channels,configs["size"],configs["size"]), 
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
        
        print("pre-trained model and configs loaded successfully")
        glow.set_actnorm_init()

    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please check the path.")
    
    kspace_trans = PseudoKSpace()

    
    for i in range(args.epochs):
        print("Epoch %d/%d"%(i+1,args.epochs))
        with torch.no_grad():
            glow(torch.randn(1, channels, args.size, args.size).to(args.device))
            z_sample, z_sample_t = glow.generate_z(n=10,mu=0,std=0.7,to_torch=True)
            x_gen_kspace = glow(z_sample_t, reverse=True)
            x_gen = kspace_trans(x_gen_kspace, reverse=True)
            # x_gen = glow.postprocess(x_gen_kspace)
            x_gen = make_grid(x_gen,nrow=int(np.sqrt(len(x_gen))))
            x_gen = x_gen.data.cpu().numpy()
            x_gen = x_gen.transpose([1,2,0])
            if x_gen.shape[-1] == 1:
                x_gen = x_gen[...,0]
            if not os.path.exists(save_path+"/samples_training"):
                os.makedirs(save_path+"/samples_training")
            x_gen = (np.clip(x_gen, 0, 1) * 255).astype("uint8")
            sio.imsave(save_path+"/samples_training/%0.6d.jpg"%i, x_gen)
 

def sampleBraTS(args):
    training_folder = f"./data/{args.dataset}_preprocessed/train/"
    save_path       = f"./samples/"
    if args.dataset == "BraTS_png":
        trans      = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize(args.size),  
                                        transforms.CenterCrop((args.size, args.size)),
                                        transforms.ToTensor()])
    elif args.dataset == "celeba":
        trans      = transforms.Compose([transforms.Resize(args.size),  
                                        transforms.CenterCrop((args.size, args.size)),
                                        transforms.ToTensor()])
    dataset    = datasets.ImageFolder(training_folder, transform=trans)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # sample 10 images from the dataset, and save them to the samples_training folder
    for i, (data, label) in enumerate(dataloader):
        if i > 10:
            break
        x_gen = data.data.cpu().numpy().squeeze(0)
        x_gen = (x_gen * 255).clip(0, 255).astype(np.uint8)
        x_gen = x_gen.transpose([1,2,0])
        if x_gen.shape[-1] == 1:
            x_gen = x_gen.squeeze(-1)
        sio.imsave(save_path+"/%0.6d.png"%i, x_gen)
        print(f"saved {i} images")


def sampleRecovered(args):
    total_dir = f"./results/{args.dataset}/{args.exp}"
    for exp_dir in os.listdir(total_dir):
        if os.path.isdir(os.path.join(total_dir, exp_dir)):
            # read the recovered.npy file
            original = np.load(os.path.join(total_dir, exp_dir, "original.npy")).squeeze(-1)
            recovered = np.load(os.path.join(total_dir, exp_dir, "recovered.npy")).squeeze(-1)
            
            # change from F mode to C mode
            original = (original * 255).clip(0, 255).astype(np.uint8)
            recovered = (recovered * 255).clip(0, 255).astype(np.uint8)

            print(f"saving recovered images to {os.path.join(total_dir, exp_dir, 'samples_recovered')}")
            
            os.makedirs(os.path.join(total_dir, exp_dir, "samples_recovered"), exist_ok=True)
            count = 0
            for i in range(len(recovered)):
                # save the recovered images and original images in a grid
                recovered_grid = np.concatenate([original[i], recovered[i]], axis=1)
                sio.imsave(os.path.join(total_dir, exp_dir, "samples_recovered", "recovered_%0.6d.png"%i), recovered_grid)
                count += 1
            print(f"saved {count} images in {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train glow network')
    parser.add_argument('-dataset',type=str,help='the dataset to train the model on', default='celeba')
    parser.add_argument('-size',type=int,help='images will be resized to this dimension',default=64)
    parser.add_argument('-epochs',type=int,help='epochs to train for',default=10)
    parser.add_argument('-device',type=str,help='whether to use',default="cpu")  
    parser.add_argument('-job_id', type=str, help='job id to save the model', default=0) 
    parser.add_argument('-exp', type=str, help='experiment name', default="1")
    args = parser.parse_args()
    # Try to initialize CUDA, fallback to CPU if fails
    # if args.device == "cuda":
    #     if not torch.cuda.is_available():
    #         raise RuntimeError("CUDA not available")
    #     # force trigger device count to check CUDA runtime
    #     torch.cuda.device_count()
    #     print("CUDA initialized successfully.")
    #     print(f"Using device: {args.device}, GPU count: {torch.cuda.device_count()}")
            
    # sampleGlow(args)
    # sampleBraTS(args)
    sampleRecovered(args)
