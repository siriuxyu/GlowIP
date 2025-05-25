import os
import nibabel as nib
import numpy as np
import torch
import cv2
import shutil

def process_nii_folder(input_dir, output_dir, target_size=(128, 128), normalize=True):
    os.makedirs(output_dir, exist_ok=True)
    global_slice_count = 0

    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for filename in os.listdir(subdir_path):
            if filename.endswith("t1.nii.gz"):
                nii_path = os.path.join(subdir_path, filename)

                base_name = filename.replace(".nii.gz", "")
                print(f"Processing {base_name}")

                img = nib.load(nii_path)
                data = img.get_fdata()

                for i in range(data.shape[-1]):
                    if i in range(70, 101) and (i % 4 == 0):
                            slice_2d = data[:, :, i]

                            # normalize for non-label images
                            if normalize:
                                slice_2d = (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d) + 1e-6) * 255
                                slice_2d = slice_2d.astype(np.uint8)

                            # resize to target size
                            slice_2d = cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_AREA)

                            # ensure the image is 2D (single channel)
                            slice_2d = slice_2d.reshape(target_size)

                            # save the slice as a PNG file with single channel
                            out_path = os.path.join(output_dir, f"slice_{global_slice_count:05d}.png")
                            cv2.imwrite(out_path, slice_2d)
                            global_slice_count += 1

                            print(f"Processed {filename}, saved to {out_path}")
                        

    print(f"Total slices processed: {global_slice_count}")


def split_train_test(input_folder, output_folder_train, output_folder_test, ratio=0.8):
    print(f"Splitting {input_folder} into {output_folder_train} and {output_folder_test} with ratio {ratio}")
    os.makedirs(output_folder_train, exist_ok=True)
    os.makedirs(output_folder_test, exist_ok=True)
    total_files_moved = 0
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.endswith(".png"):
            if np.random.rand() < ratio:
                shutil.move(file_path, os.path.join(output_folder_train, filename))
            else:
                shutil.move(file_path, os.path.join(output_folder_test, filename))
            print(f"Moved {filename} to {output_folder_train if np.random.rand() < ratio else output_folder_test}")
            total_files_moved += 1
    print(f"Total files moved: {total_files_moved}")
        

def sample_images(dir, num_samples=10):
    num_files = len(os.listdir(dir))
    sample_indices = np.random.choice(num_files, num_samples, replace=False)
    # Create a grid of 5x2 images with space for title
    title_height = 50  # Height for title
    grid = np.zeros((title_height + 128 * 2, 128 * 5, 3), dtype=np.uint8)
    grid.fill(0)  # White background

    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Sampled BraTS Images"
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = (grid.shape[1] - text_size[0]) // 2  # Center the text
    text_y = 30  # Position from top
    cv2.putText(grid, title, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    images = []
    for i, index in enumerate(sample_indices):
        filename = os.listdir(dir)[index]
        print(f"Sample {i+1}: {filename}")
        img = cv2.imread(os.path.join(dir, filename))
        if img is None:
            print(f"Error loading image {filename}")
            continue
            
        img = cv2.resize(img, (128, 128))
        images.append(img)
        
        # Calculate position in grid
        row = i // 5
        col = i % 5
        y_start = title_height + row * 128
        x_start = col * 128
        
        # Place the image in the grid
        grid[y_start:y_start+128, x_start:x_start+128] = img

    # Save the grid
    os.makedirs("../images", exist_ok=True)
    cv2.imwrite(f"../images/sample_images_{num_samples}.png", grid)
    print(f"Grid saved as ../images/sample_images_{num_samples}.png")
    
    images = np.array(images)
    return images

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


# process_nii_folder("BraTS", "BraTS_png_preprocessed", target_size=(128, 128), normalize=True)

# split_train_test("BraTS_png_preprocessed", "BraTS_png_preprocessed/train/train", "BraTS_png_preprocessed/test/test", ratio=0.8)

trans = PseudoKSpace()
imgs = sample_images("BraTS_png_preprocessed/train/train", num_samples=10)
imgs = torch.from_numpy(imgs).float()
# convert to grayscale
imgs = imgs.permute(0, 3, 1, 2).contiguous()  # (B,H,W,C) -> (B,C,H,W)
if imgs.ndim == 4 and imgs.shape[1] == 3:
    imgs = imgs.mean(dim=1, keepdim=True)  # (B,C,H,W) -> (B,1,H,W)
print(f"Shape of original images: {imgs.shape}")

k_space = trans(imgs, reverse=False)
print(f"Shape of k-space: {k_space.shape}")

imgs_reconstructed = trans(k_space, reverse=True)
print(f"Shape of reconstructed images: {imgs_reconstructed.shape}")

if imgs.shape[1] == 1:
    imgs_reconstructed = imgs_reconstructed.squeeze(1)  # (B,1,H,W) -> (B,H,W)

# Save the reconstructed images
for i, img in enumerate(imgs_reconstructed):
    # save original image and reconstructed image as a grid
    grid = np.zeros((128, 128 * 2), dtype=np.uint8)
    grid.fill(0)  # White background
    img = img.detach().numpy()
    
    grid[:, :128] = img
    grid[:, 128:] = imgs[i].numpy().squeeze(0)  # Original image
    # Save the image
    out_path = os.path.join("../images/k_space", f"reconstructed_{i}.png")
    cv2.imwrite(out_path, grid)
    print(f"Reconstructed image {i} saved.")