import os
import nibabel as nib
import numpy as np
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

    for i, index in enumerate(sample_indices):
        filename = os.listdir(dir)[index]
        print(f"Sample {i+1}: {filename}")
        img = cv2.imread(os.path.join(dir, filename))
        if img is None:
            print(f"Error loading image {filename}")
            continue
            
        img = cv2.resize(img, (128, 128))
        
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

# process_nii_folder("BraTS", "BraTS_png_preprocessed", target_size=(128, 128), normalize=True)

# split_train_test("BraTS_png_preprocessed", "BraTS_png_preprocessed/train/train", "BraTS_png_preprocessed/test/test", ratio=0.8)

sample_images("BraTS_png_preprocessed/train/train", num_samples=10)

