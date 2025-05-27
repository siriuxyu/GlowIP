import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse
import skimage.io as sio
from glow.glow import Glow
from .lasso_utils import celebA_estimators
import json
import os
from glob import glob
import easydict
from scipy.linalg import null_space
import cv2
from collections import OrderedDict

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def GlowCSK(args):
    '''
    This function is used to solve the CS problem with Glow model and forward FFT.
    z -> x -> x_k
    x_k computer loss with x_k_groundtruth
    '''
    if args.init_norms == None:
        args.init_norms = [None]*len(args.m)
    else:
        assert args.init_strategy == "random_fixed_norm", "init_strategy should be random_fixed_norm if init_norms is used"
    assert len(args.m) == len(args.gamma) == len(args.init_norms), "length of either m, gamma or init_norms are not same"
    loopOver = zip(args.m, args.gamma, args.init_norms)
    
    if args.dataset == "celeba":
        channels = 3
    elif args.dataset == "BraTS" or args.dataset == "BraTS_png":
        channels = 1
    else:
        raise "dataset not defined"

    # loading dataset and sampling images before the loop
    npz_file = f"./data/test_data/BraTS/BraTS_test.npz"
    test_dataset = NPZDataset(npz_file, size=args.size)
    test_dataset.sample_images(args.batchsize * 4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, drop_last=True, shuffle=False)

    for m, gamma, init_norm in loopOver:
        skip_to_next = False # flag to skip to next loop if recovery is fails due to instability
        n                  = args.size * args.size * 1      # 1 channel for MRI
        modeldir           = f"./trained_models/{args.dataset}/glow_{args.size}_{args.job_id}/"
        save_path          = f"./results/{args.dataset}_{args.size}/{args.experiment}"

        # loading glow configurations
        config_path = modeldir + "configs.json"
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        # sensing matrix
        mask = np.zeros(n)
        mask[np.random.choice(n,m,replace=False)] = 1
        mask = torch.tensor(mask,dtype=torch.float, requires_grad=False, device=args.device)
        A = mask.to(torch.complex64).view(1,n) 
        
        # regularizor
        gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
    
        # adding noise
        if  args.noise == "random_bora":
            noise = np.random.normal(0,1,size=(args.batchsize,m))
            noise = noise * 0.1/np.sqrt(m)
            noise = torch.tensor(noise,dtype=torch.float,requires_grad=False, device=args.device)
        else:
            noise = np.random.normal(0,1,size=(args.batchsize,m))
            noise = noise / (np.linalg.norm(noise,2,axis=-1, keepdims=True)) * float(args.noise)
            noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=args.device)
        
        
        ########## start solving over batches ##########
        Original = []; Original_k = []; 
        Compressed_k = [];
        Recovered = []; Recovered_k = [];
        Z_Recovered = [];  
        Residual_Curve = []; Recorded_Z = []; 
        for i, data in enumerate(test_dataloader):
            x_test = data   # x_test = data[0] for other datasets (besides npz)
            x_test_k = img_to_k(x_test)
            x_test = x_test.clone().to(device=args.device)
            x_test_k = x_test_k.clone().to(device=args.device)
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"
            
                
            # loading glow model
            glow = Glow((channels,args.size,args.size),    # 1 channel for MRI
                        K=configs["K"],L=configs["L"],
                        coupling=configs["coupling"],
                        n_bits_x=configs["n_bits_x"],
                        nn_init_last_zeros=configs["last_zeros"],
                        device=args.device)
            
            state_dict = torch.load(modeldir + "glowmodel.pt")
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")  # 去掉前缀
                new_state_dict[new_key] = v
                
            glow.load_state_dict(new_state_dict)
            
            glow.eval()
            
            if torch.cuda.device_count() > 1:
                glow = torch.nn.DataParallel(glow)
            
            core_glow = glow.module if isinstance(glow, torch.nn.DataParallel) else glow
            
            # to avoid lazy initialization of actnorm
            with torch.no_grad():
                core_glow(torch.randn(1, channels, args.size, args.size).to(args.device))
            
            # making a forward to record shapes of z's for reverse pass
            _ = glow(core_glow.preprocess(torch.zeros_like(x_test)))
            
            z_sampled = sample_z(glow, core_glow, args, n_test, n, m=m, init_norm=init_norm, A=A, noise=noise, x_test=x_test)
            
            # selecting optimizer
            if args.optim == "adam":
                optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
            elif args.optim == "lbfgs":
                optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
            else:
                raise "optimizer not defined"
            
            # to be recorded over iteration
            psnr_t    = torch.nn.MSELoss().to(device=args.device)
            residual  = []; recorded_z = []
            # running optimizer steps
            for t in range(args.steps):
                def closure():
                    optimizer.zero_grad()
                    z_unflat    = core_glow.unflatten_z(z_sampled, clone=False)
                    x_gen       = glow(z_unflat, reverse=True, reverse_clone=False)
                    x_gen       = core_glow.postprocess(x_gen,floor_clamp=False)
                    x_gen_k     = img_to_k(x_gen)
                    x_test_k_flat = x_test_k.view([-1,n])
                    x_gen_k_flat  = x_gen_k.view([-1,n])
                    y_true_k      = x_test_k_flat * A       # + noise
                    y_gen_k       = x_gen_k_flat * A
                    global residual_t
                    delta = y_gen_k - y_true_k
                    residual_t = (delta.real ** 2 + delta.imag ** 2).sum(dim=1).mean()
                    if args.z_penalty_unsquared:
                        z_reg_loss_t= gamma*(z_sampled.norm(dim=1)**2).mean()
                    else:
                        z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
                    loss_t      = residual_t + z_reg_loss_t
                    psnr        = psnr_t(x_test, x_gen)
                    psnr        = 10 * np.log10(1 / psnr.item())
                    print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f"%(t,loss_t.item(),residual_t.item(),z_reg_loss_t.item(), psnr),end="\r")
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
                    skip_to_next = True
                    break
            
            if skip_to_next:
                break
            
            # getting recovered and true images
            with torch.no_grad():
                x_test_k_flat = x_test_k.view([-1,n]) 
                y_test_k = x_test_k_flat * A
                x_test_np = x_test.data.cpu().numpy().transpose(0,2,3,1)
                x_test_k_np = x_test_k.data.cpu().numpy().transpose(0,2,3,1)
                y_test_k_np = y_test_k.data.cpu().numpy()
                z_unflat  = core_glow.unflatten_z(z_sampled, clone=False)
                x_gen     = glow(z_unflat, reverse=True, reverse_clone=False)
                x_gen     = core_glow.postprocess(x_gen,floor_clamp=False)
                x_gen_k   = img_to_k(x_gen)
                x_gen_np  = x_gen.data.cpu().numpy().transpose(0,2,3,1)
                x_gen_k_np  = x_gen_k.data.cpu().numpy().transpose(0,2,3,1)
                x_gen_np  = np.clip(x_gen_np,0,1)
                z_recov   = z_sampled.data.cpu().numpy()
            
            Original.append(x_test_np)
            Original_k.append(x_test_k_np)
            Compressed_k.append(y_test_k_np)
            Recovered.append(x_gen_np)
            Recovered_k.append(x_gen_k_np)
            Z_Recovered.append(z_recov)
            Residual_Curve.append(residual)
            Recorded_Z.append(recorded_z)
                    
            # freeing up memory for second loop
            core_glow.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, glow
            torch.cuda.empty_cache()
            print("\nbatch completed")
        
        if skip_to_next:
            print("\nskipping current loop due to instability or user triggered quit")
            continue
    
        # collecting everything together 
        Original     = np.vstack(Original)
        Original_k   = np.vstack(Original_k)
        Compressed_k = np.vstack(Compressed_k)
        Recovered    = np.vstack(Recovered)
        Recovered_k  = np.vstack(Recovered_k)
        Z_Recovered  = np.vstack(Z_Recovered)
        Recorded_Z   = np.vstack(Recorded_Z) 

        # Calculate all metrics
        psnr_list = []
        ssim_list = []
        nrmse_list = []
        
        for x, y in zip(Original, Recovered):
            # For multi-channel images, calculate metrics for each channel and average
            if x.shape[-1] > 1:  # Multi-channel image
                psnr_channels = []
                ssim_channels = []
                nrmse_channels = []
                for c in range(x.shape[-1]):
                    psnr_channels.append(compare_psnr(x[..., c], y[..., c]))
                    ssim_channels.append(compare_ssim(x[..., c], y[..., c], data_range=1.0))
                    nrmse_channels.append(compare_nrmse(x[..., c], y[..., c], normalization='min-max'))
                psnr_list.append(np.mean(psnr_channels))
                ssim_list.append(np.mean(ssim_channels))
                nrmse_list.append(np.mean(nrmse_channels))
            else:  # Single channel image
                psnr_list.append(compare_psnr(x, y))
                ssim_list.append(compare_ssim(x, y, data_range=1.0))
                nrmse_list.append(compare_nrmse(x, y, normalization='min-max'))
        
        psnr = np.array(psnr_list)
        ssim = np.array(ssim_list)
        nrmse = np.array(nrmse_list)
        z_recov_norm = np.linalg.norm(Z_Recovered, axis=-1)
        
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test        = %d\n"%len(Recovered)
        printout = printout + "\t n             = %d\n"%n
        printout = printout + "\t m             = %d\n"%m
        printout = printout + "\t gamma         = %0.6f\n"%gamma
        printout = printout + "\t optimizer     = %s\n"%args.optim
        printout = printout + "\t lr            = %0.3f\n"%args.lr
        printout = printout + "\t steps         = %0.3f\n"%args.steps
        printout = printout + "\t init_strategy = %s\n"%args.init_strategy
        printout = printout + "\t init_std      = %0.3f\n"%args.init_std
        if init_norm is not None:
            printout = printout + "\t init_norm     = %0.3f\n"%init_norm
        printout = printout + "\t z_recov_norm  = %0.3f\n"%np.mean(z_recov_norm)
        printout = printout + "\t PSNR          = %0.3f ± %0.3f\n"%(np.mean(psnr), np.std(psnr))
        printout = printout + "\t SSIM          = %0.3f ± %0.3f\n"%(np.mean(ssim), np.std(ssim))
        printout = printout + "\t NRMSE         = %0.3f ± %0.3f\n"%(np.mean(nrmse), np.std(nrmse))
        print(printout)
        
        # saving printout
        if args.save_metrics_text:
            with open("%s_cs_glow_results_%s.txt"%(args.dataset,args.experiment),"a") as f:
                f.write('\n' + printout)
    
        
        # setting folder to save results in 
        if args.save_results:
            gamma = gamma.item()
            file_names = np.arange(len(Recovered))
            # file_names = [name[0].split("/")[-1] for name in test_dataset.samples]
            if args.init_strategy == "random":
                save_path_template = save_path + "/cs_m_%d_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"
                save_path = save_path_template%(m,gamma,args.steps,args.lr,args.init_std,args.optim)
            elif args.init_strategy == "random_fixed_norm":
                save_path_template = save_path+"/cs_m_%d_gamma_%0.6f_steps_%d_lr_%0.3f_init_%s_%0.3f_optim_%s"
                save_path          = save_path_template%(m,gamma,args.steps,args.lr,args.init_strategy,init_norm, args.optim)
            else:
                save_path_template = save_path + "/cs_m_%d_gamma_%0.6f_steps_%d_lr_%0.3f_init_%s_optim_%s"
                save_path          = save_path_template%(m,gamma,args.steps,args.lr,args.init_strategy,args.optim)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                save_path_1 = save_path + "_1"
                if not os.path.exists(save_path_1):
                    os.makedirs(save_path_1)
                    save_path = save_path_1
                else:
                    save_path_2 = save_path + "_2"
                    if not os.path.exists(save_path_2):
                        os.makedirs(save_path_2)
                        save_path = save_path_2
            print(f"Saving results to {save_path}")
            # saving results now
            try:
                _ = [sio.imsave(save_path+"/"+str(name), x) for x,name in zip(Recovered,file_names)]
            except Exception as e:
                print(f"\n[ERROR] Saving results failed due to: {e}")
            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            np.save(save_path+"/original.npy", Original)
            np.save(save_path+"/recovered.npy", Recovered)
            np.save(save_path+"/z_recovered.npy", Z_Recovered)
            np.save(save_path+"/residual_curve.npy", Residual_Curve)
            np.save(save_path+"/metrics.npy", {
                'psnr': psnr,
                'ssim': ssim,
                'nrmse': nrmse,
                'z_recov_norm': z_recov_norm
            })
        torch.cuda.empty_cache()
        
        
def img_to_k(img):
    """
    from MRI real image to k-space complex data
    img: (B, 1, H, W), Tensor, real image
    return: (B, 1, H, W), Tensor, complex data
    """
    x = img.to(torch.complex64)
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
    