import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse
import skimage.io as sio
from glow.glow import Glow
import json
import os
import cv2
from collections import OrderedDict
import argparse

        
        
def img_to_k(img):
    """
    img: (B, 1, H, W), Tensor, real image
    return: (B, 1, H, W), Tensor, complex data
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Ensure input is float32
    x = img.float()
    # Convert to complex
    x = x.to(torch.complex64)
    # Perform FFT
    kspace_x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    return kspace_x



class NPZDataset(Dataset):
    def __init__(self, npz_file_path, size=64):
        data = np.load(npz_file_path)

        self.images = data['all_imgs']
        self.length = len(self.images)
        self.size = size
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            # GrayScale: H x W → 1 x H x W
            img = img[np.newaxis, :, :]
        elif img.ndim == 3 and img.shape[0] not in (1, 3):
            # Color: C x H x W → 1 x C x H x W
            img = np.transpose(img, (2, 0, 1))
            
        img_resized = np.zeros((img.shape[0], self.size, self.size), dtype=np.float32)
        for c in range(img.shape[0]):
            img_resized[c] = cv2.resize(img[c], (self.size, self.size), interpolation=cv2.INTER_AREA)

        return torch.from_numpy(img_resized / 255.0).float()

    def sample_images(self, num_samples=1000):
        """
        Randomly sample a specified number of images from the dataset.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        elif num_samples >= self.length:
            num_samples = self.length
            return
        else:
            indices = np.random.choice(self.length, num_samples, replace=False)
            sampled_images = self.images[indices]
            self.images = sampled_images
            self.length = num_samples


def sample_z(glow, core_glow, args, n_test, n, channels=1, m=None, init_norm=None, A=None, noise=None, x_test=None):

    # initializing z from Gaussian with std equal to init_std
    if args.init_strategy == "random":
        z_sampled = np.random.normal(0,args.init_std,[n_test,n])
        z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
    # intializing z from Gaussian and scaling its norm to init_norm
    elif args.init_strategy == "random_fixed_norm":
        z_sampled = np.random.normal(0,1,[n_test,n])
        z_sampled = z_sampled / np.linalg.norm(z_sampled, axis=-1, keepdims=True)
        z_sampled = z_sampled * init_norm
        z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
        print("z intialized with a norm equal to = %0.1f"%init_norm)
    else:
        raise "Initialization strategy not defined"
    
    return z_sampled




def GlowCSK_single(args):
    m_new = []
    down_sample_rates = [2, 3, 4, 8]
    n = args.size * args.size
    for down_sample_rate in down_sample_rates:
        m_new.append(n // down_sample_rate)
    args.m = m_new
    args.gamma = [0] * len(m_new)
    args.device = "cpu"
    img = "./data/test_data/BraTS/images/921.png"


    # load and convert to greyscale, (1, 1, H, W)
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, :, :]
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    print(f"img.shape: {img.shape}")

    # loading glow configurations
    modeldir = "./trained_models/BraTS_png/glow_128_0/"
    config_path = modeldir + "configs.json"
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    # loading glow model
    glow = Glow((1,args.size,args.size),    # 1 channel for MRI
                K=configs["K"],L=configs["L"],
                coupling=configs["coupling"],
                n_bits_x=configs["n_bits_x"],
                nn_init_last_zeros=configs["last_zeros"],
                device=args.device)

    # load to cpu
    state_dict = torch.load(modeldir + "glowmodel.pt", map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # 去掉前缀
        new_state_dict[new_key] = v

    glow.load_state_dict(new_state_dict)
    glow.eval()

    # initialize glow.sizes
    glow(torch.randn(1,1,args.size,args.size))

    # Loop over all m values
    for i, cs_rate in enumerate(down_sample_rates):
        m = args.m[i]

        # sensing matrix
        mask = np.zeros(n)
        mask[np.random.choice(n,m,replace=False)] = 1
        mask = torch.tensor(mask, dtype=torch.float, requires_grad=False, device=args.device)
        F = mask.to(torch.complex64).view(1,n) 

        x_test = img.clone()
        # Ensure x_test is float32
        x_test = x_test.float()
        x_test_k = img_to_k(x_test)
        x_test_k_flat = x_test_k.view([-1,n])
        y_test_k = x_test_k_flat * F

        print(f"+-+-+-+-+-+-+ down_sample_rate: {cs_rate}, m: {m} +-+-+-+-+-+-+")
        # Generate z
        z_sampled = np.random.normal(0,args.init_std,[1,n])
        z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
        print(f"z_sampled.dtype: {z_sampled.dtype}")
        # selecting optimizer
        if args.optim == "adam":
            optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
        elif args.optim == "lbfgs":
            optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
        else:
            raise "optimizer not defined"
        residual = []
        recorded_z = []
        for t in range(args.steps):
            def closure():
                optimizer.zero_grad()
                z_unflat    = glow.unflatten_z(z_sampled, clone=False)
                x_gen       = glow(z_unflat, reverse=True, reverse_clone=False)
                x_gen       = glow.postprocess(x_gen,floor_clamp=False)

                x_gen_k     = img_to_k(x_gen)
                x_gen_k_flat  = x_gen_k.view([-1,n])
                y_gen_k       = x_gen_k_flat * F

                global residual_t
                delta = y_gen_k - y_test_k
                residual_t = (delta.real ** 2 + delta.imag ** 2).sum(dim=1).mean()
                # if args.z_penalty_unsquared:
                #     z_reg_loss_t= gamma*(z_sampled.norm(dim=1)**2).mean()
                # else:
                #     z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
                loss_t      = residual_t    # + z_reg_loss_t
                psnr        = compare_psnr(x_test, x_gen)
                psnr        = 10 * np.log10(1 / psnr.item())
                print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f"%(t,loss_t.item(),residual_t.item(), psnr),end="\r")
                
                if args.optim == "lbfgs":
                    loss_t.backward(retain_graph=True)
                else:
                    loss_t.backward()

                return loss_t
            try:
                optimizer.step(closure)
                recorded_z.append(z_sampled.data.cpu().numpy())
                residual.append(residual_t.item())
            except Exception as e:
                print(f"\n[ERROR] Step failed due to: {e}")
                break

        with torch.no_grad():
            z_recov   = z_sampled.data.cpu().numpy()
            z_unflat  = glow.unflatten_z(z_sampled, clone=False)
            x_gen     = glow(z_unflat, reverse=True, reverse_clone=False)
            x_gen_np  = x_gen.data.cpu().numpy().squeeze(1)
            x_gen_np  = np.clip(x_gen_np,0,1)
            x_test_np = x_test.data.cpu().numpy().squeeze(1)
            x_test_np = np.clip(x_test_np,0,1)

            residual = np.array(residual)
            recorded_z = np.array(recorded_z)

            psnr = compare_psnr(x_test_np, x_gen_np)
            psnr = 10 * np.log10(1 / psnr.item())
            print(f"psnr: {psnr}")

            ssim = compare_ssim(x_test_np, x_gen_np, data_range=1.0)
            print(f"ssim: {ssim}")

            nrmse = compare_nrmse(x_test_np, x_gen_np)
            print(f"nrmse: {nrmse}")

        os.makedirs(f"results/local", exist_ok=True)
        np.save(f"results/local/cs_glow_local_{cs_rate}.npy", {
            'z_recov': z_recov,
            'x_gen_np': x_gen_np,
            'x_test_np': x_test_np,
            'residual': residual,
            'recorded_z': recorded_z,
        })

        

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='solve compressive sensing')
    parser.add_argument('-prior',type=str,help='choose with prior to use glow, dcgan, wavelet, dct', default='glow')
    parser.add_argument('-experiment', type=str, help='the name of the experiment',default='celeba_cs_glow')
    parser.add_argument('-dataset', type=str, help='the dataset/images to use',default='celeba')
    parser.add_argument('-model', type=str, help='which model to use',default='celeba')
    parser.add_argument('-m',  type=int, nargs='+',help='no. of measurements',default=[12288,10000,7500,5000,2500,1000,750,500,400,300,200,100,50,30,20])
    parser.add_argument('-gamma',  type=float, nargs='+',help='regularizor',default=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    parser.add_argument('-optim', type=str, help='optimizer', default="lbfgs")
    parser.add_argument('-lr', type=float, help='learning rate', default=0.02)
    parser.add_argument('-steps',type=int,help='no. of steps to run', default=50)
    parser.add_argument('-batchsize',type=int, help='no. of images to solve in parallel as batches',default=6)
    parser.add_argument('-size',type=int, help='resize all images to this size', default=128)
    parser.add_argument('-device',type=str,help='device to use', default='cuda')
    parser.add_argument('-noise',type=str, help='noise to add. Either random_bora or float representing std of gaussian noise', default="random_bora")
    parser.add_argument('-init_strategy',type=str,help="init strategy to use",default='random')
    parser.add_argument('-init_std', type=float,help='std of init_strategy is random', default=0)
    parser.add_argument('-init_norms', type=float, nargs='+',help='initialization norm',default=None)
    parser.add_argument('-save_metrics_text',type=bool, help='whether to save results to a text file',default=True)
    parser.add_argument('-save_results',type=bool,help='whether to save results after experiments conclude',default=True)
    parser.add_argument('-z_penalty_unsquared', action="store_true",help="use ||z|| if True else ||z||^2")
    parser.add_argument('-job_id', type=str, help='job id to use for logging', default='0')
    args = parser.parse_args()

    # GlowCSK(args)
    # Test_GlowCSK(args)
    GlowCSK_single(args)
    
    




# def Test_GlowCSK(args):
#     args.device = "cpu"
#     size = 128
#     m = 5000
#     n = size * size
#     npz_file = "./data/test_data/BraTS/BraTS_test.npz"
#     # loading dataset
#     test_dataset    = NPZDataset(npz_file, size=size)
#     test_dataset.sample_images(8)
#     test_dataloader = DataLoader(test_dataset,batch_size=4,drop_last=True,shuffle=False)

#     # sensing matrix
#     mask = np.zeros(n)
#     # choose m dp to be 1, the rest to be 0
#     mask[np.random.choice(n,m,replace=False)] = 1
#     mask = torch.tensor(mask,dtype=torch.float, requires_grad=False, device=args.device)
#     F = mask.to(torch.complex64).view(1,n) 

#     # loading glow configurations
#     modeldir = "./trained_models/BraTS_png/glow_128_0/"
#     config_path = modeldir + "configs.json"
#     with open(config_path, 'r') as f:
#         configs = json.load(f)

#     # loading glow model
#     glow = Glow((1,size,size),    # 1 channel for MRI
#                 K=configs["K"],L=configs["L"],
#                 coupling=configs["coupling"],
#                 n_bits_x=configs["n_bits_x"],
#                 nn_init_last_zeros=configs["last_zeros"],
#                 device=args.device)

#     # load to cpu
#     state_dict = torch.load(modeldir + "glowmodel.pt", map_location=torch.device('cpu'))
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         new_key = k.replace("module.", "")  # 去掉前缀
#         new_state_dict[new_key] = v

#     glow.load_state_dict(new_state_dict)
#     glow.eval()

#     # initialize glow.sizes
#     glow(torch.randn(1,1,size,size))

#     Original = []
#     Recovered = []
    

#     for i, data in enumerate(test_dataloader):
#         x_test = data
#         x_test_k = img_to_k(x_test)
#         x_test_k = x_test_k.view([-1,n])
#         y_test_k = x_test_k * F


#         # z_sampled = np.random.normal(0,args.init_std,[n_test,n])
#         z_sampled = np.random.normal(0,0.1,[4,n])
#         z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
#         z_unflat  = glow.unflatten_z(z_sampled, clone=False)
#         x_gen     = glow(z_unflat, reverse=True, reverse_clone=False)
#         x_gen     = glow.postprocess(x_gen,floor_clamp=False)
#         print(x_gen.shape) 
#         x_gen_k   = img_to_k(x_gen)
#         print(x_gen_k.shape)
#         x_gen_k_flat = x_gen_k.view([-1,n])
#         y_gen_k = x_gen_k_flat * F

#         x_gen_np = x_gen.data.cpu().numpy().squeeze(1)
#         x_test_np = x_test.data.cpu().numpy().squeeze(1)
#         Original.append(x_test_np)
#         Recovered.append(x_gen_np)

#     Original = np.vstack(Original)
#     Recovered = np.vstack(Recovered)
#     print(Original.shape)
#     print(Recovered.shape)

#     psnr_list = []
#     ssim_list = []
#     nrmse_list = []
#     for x, y in zip(Original, Recovered):
#     # For multi-channel images, calculate metrics for each channel and average
#         if (len(x.shape) > 2) and (x.shape[0] > 1):
#             psnr_channels = []
#             ssim_channels = []
#             nrmse_channels = []
#             for c in range(x.shape[0]):
#                 psnr_channels.append(compare_psnr(x[:, c, :, :], y[:, c, :, :]))
#                 ssim_channels.append(compare_ssim(x[:, c, :, :], y[:, c, :, :], data_range=1.0))
#                 nrmse_channels.append(compare_nrmse(x[:, c, :, :], y[:, c, :, :], normalization='min-max'))
#             psnr_list.append(np.mean(psnr_channels))
#             ssim_list.append(np.mean(ssim_channels))
#             nrmse_list.append(np.mean(nrmse_channels))
#         else:  # Single channel image
#             psnr_list.append(compare_psnr(x, y))
#             ssim_list.append(compare_ssim(x, y, data_range=1.0))
#             nrmse_list.append(compare_nrmse(x, y, normalization='min-max'))
    
#     psnr = np.array(psnr_list)
#     ssim = np.array(ssim_list)
#     nrmse = np.array(nrmse_list)
#     print(f"psnr: {psnr.mean()}, ssim: {ssim.mean()}, nrmse: {nrmse.mean()}")
#     print(f"psnr: {psnr.shape}, ssim: {ssim.shape}, nrmse: {nrmse.shape}")  

#     np.save("metrics.npy", {
#                 'psnr': psnr,
#                 'ssim': ssim,
#                 'nrmse': nrmse,
#             })
#     metrics = np.load("metrics.npy", allow_pickle=True).item()
#     print(metrics)