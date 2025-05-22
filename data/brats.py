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
        


process_nii_folder("BraTS", "BraTS_png_preprocessed", target_size=(128, 128), normalize=True)

split_train_test("BraTS_png_preprocessed", "BraTS_png_preprocessed/train/train", "BraTS_png_preprocessed/test/test", ratio=0.8)
