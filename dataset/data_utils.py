import os
import torch
from torchvision.io import read_image
from tqdm import tqdm
import torch.nn.functional as F
import spectral.io.envi as envi
import numpy as np
import csv
import yaml
import pandas as pd




# Start Project structur
def create_project_structure(config_path: str):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_name = config["project"]["project_name"]
    base_path = config["project"]["base_path"]
    shape_h = config["data"]["shape"]['h']
    project_root = os.path.join(base_path, project_name)

    RGB_HSI = ["RGB", "HSI"]

    subfolders = ["patches", "before_csv", "csv_for_dataloader"]
    splits = ["train", "val", "test"]
    
    formats = config["patch_formats"]

    for rgb_hsi in RGB_HSI :
        for subfolder in subfolders:
            for split in splits:
                for fmt in formats:

                    n_lines = fmt['n_lines']
                    patch_size = fmt['patch_size'] if fmt['patch_size'] != 0 else shape_h  # full width if 0
                    stride_lines = fmt['stride_lines']
                    stride_cols = fmt['stride_cols'] if fmt['stride_cols'] != 0 else shape_h
                    format_name = f"w{n_lines}_h{patch_size}_sl{stride_lines}_sc{stride_cols}"

                    path = os.path.join(project_root, rgb_hsi, subfolder, split, format_name)
                    os.makedirs(path, exist_ok=True)

                    # if subfolder == 'before_csv' :
                    #     df = pd.DataFrame(columns=['path_dat', 'path_npy', 'batch_size', 'C', 'H', 'W', 'label'])
                    #     path = os.path.join(path, split + '.csv')
                    #     df.to_csv(path, index=False)

                    
    print(f"Project structur : Done")
# End Project structur



# Start Patch Extraction
def extract_patches_HSI(cube, mask=None, n_lines=100, patch_size=100, stride_lines=100, stride_cols=100, full_width=False):
    H, W, C = cube.shape

    if full_width:
        patch_size = W
        stride_cols = W

    # ----------- PATCHES CUBE -----------
    x = cube.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    unfolded = torch.nn.functional.unfold(
        x,
        kernel_size=(n_lines, patch_size),
        stride=(stride_lines, stride_cols)
    )
    N = unfolded.shape[-1]
    patches = unfolded.transpose(1, 2).reshape(N, C, n_lines, patch_size)

    # ----------- OPTIONAL: PATCHES MASK & LABELS -----------
    if mask is not None:
        m = mask.unsqueeze(0).unsqueeze(0).float()
        mask_unfolded = torch.nn.functional.unfold(
            m,
            kernel_size=(n_lines, patch_size),
            stride=(stride_lines, stride_cols)
        )
        mask_patches = mask_unfolded.squeeze(0).transpose(0, 1).reshape(N, n_lines, patch_size)

        labels = (mask_patches == 255).any(dim=(1, 2)).int().tolist()
        return patches, labels

    return patches
# End Patch Extraction

def extract_patches_RGB(image, mask=None, n_lines=100, patch_size=100, stride_lines=100, stride_cols=100, full_width=False):
    """
    image: torch.Tensor de forme (3, H, W) — image RGB lue avec torchvision.io.read_image
    mask: torch.Tensor de forme (H, W) ou None
    """
    if image.shape[0] != 3:
        raise ValueError(f"Expected image shape (3, H, W), but got {image.shape}")

    image = image.float()
    C, H, W = image.shape

    if full_width:
        patch_size = W
        stride_cols = W

    # Convert to (1, C, H, W)
    x = image.unsqueeze(0).contiguous()

    # Vérifie que le patch tient dans l'image
    if H < n_lines or W < patch_size:
        raise ValueError(f"Patch size ({n_lines}, {patch_size}) is too big for image size ({H}, {W})")

    # Extraction des patches
    unfolded = torch.nn.functional.unfold(
        x,
        kernel_size=(n_lines, patch_size),
        stride=(stride_lines, stride_cols)
    )
    N = unfolded.shape[-1]
    patches = unfolded.transpose(1, 2).reshape(N, C, n_lines, patch_size)

    # ----------- OPTIONAL: PATCHES MASK & LABELS -----------
    if mask is not None:
        if mask.shape != (H, W):
            raise ValueError(f"Mask shape must be ({H}, {W}), but got {mask.shape}")
        m = mask.unsqueeze(0).unsqueeze(0).float()
        mask_unfolded = torch.nn.functional.unfold(
            m,
            kernel_size=(n_lines, patch_size),
            stride=(stride_lines, stride_cols)
        )
        mask_patches = mask_unfolded.squeeze(0).transpose(0, 1).reshape(N, n_lines, patch_size)

        # Exemple de critère : au moins un pixel avec valeur 255
        labels = (mask_patches == 255).any(dim=(1, 2)).int().tolist()
        return patches, labels

    return patches



# Start Save Patches Batch
def save_patches_batch(patches, labels, output_dir_patches, file_name, output_dir_beforcsv, split):

    normal_patches = [patch for patch, label in zip(patches, labels) if label == 0]
    anormal_patches = [patch for patch, label in zip(patches, labels) if label == 1]

    def save_memmap(patches_list, label_val, subdir):
        if not patches_list:
            return

        patches_tensor = torch.stack(patches_list)
        patches_numpy = patches_tensor.cpu().numpy()

        dir_path = os.path.join(output_dir_patches, subdir)
        os.makedirs(dir_path, exist_ok=True)

        patches_path = os.path.join(dir_path, f"{file_name}_patches.dat")
        shape_path = os.path.join(dir_path, f"{file_name}_shape.npy")

        patches_memmap = np.memmap(patches_path, dtype='float32', mode='w+', shape=patches_numpy.shape)
        patches_memmap[:] = patches_numpy[:]
        np.save(shape_path, patches_numpy.shape)

        if split == 'train':
            dir_path_beforcsv = os.path.join(output_dir_beforcsv, 'train_' + os.path.basename(output_dir_beforcsv) + '.csv')
        elif split == 'test' :
            dir_path_beforcsv = os.path.join(output_dir_beforcsv, 'test_' + os.path.basename(output_dir_beforcsv) + '.csv')
        elif split == 'val' :
            dir_path_beforcsv = os.path.join(output_dir_beforcsv, 'val_' + os.path.basename(output_dir_beforcsv) + '.csv')

        # Ajouter dans le CSV
        B, C, H, W = patches_numpy.shape
        fieldnames = ['path_dat', 'path_npy', 'batch_size', 'C', 'H', 'W', 'label']
        file_exists = os.path.exists(dir_path_beforcsv)
        write_header = not file_exists or os.stat(dir_path_beforcsv).st_size == 0

        with open(dir_path_beforcsv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'path_dat': os.path.abspath(patches_path),
                'path_npy': os.path.abspath(shape_path),
                'batch_size': B,
                'C': C,
                'H': H,
                'W': W,
                'label': label_val
            })

        # with open(dir_path_beforcsv, mode='a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=['path_dat', 'path_npy', 'batch_size', 'C', 'H', 'W', 'label'])
        #     writer.writerow({
        #         'path_dat': os.path.abspath(patches_path),
        #         'path_npy': os.path.abspath(shape_path),
        #         'batch_size': B,
        #         'C': C,
        #         'H': H,
        #         'W': W,
        #         'label': label_val
        #     })

    save_memmap(normal_patches, 0, "normal")
    save_memmap(anormal_patches, 1, "anormal")
# End Save Patches Batch

# Start Load HSI Data
def get_cube(hdrfile):
    img = envi.open(hdrfile)
    info = envi.read_envi_header(hdrfile)
    if "reflectance scale factor" in info:
        img = img.asarray() / float(info["reflectance scale factor"])
    return img, [float(v) for v in info['wavelength']]
# End Load HSI Data


def get_sorted_files(path, ext):
    return sorted([f for f in os.listdir(path) if f.endswith(ext)])


# Start csv_for_dataloader
def csv_for_dataloader(config_path: str):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_name = config["project"]["project_name"]
    base_path = config["project"]["base_path"]
    shape_h = config["data"]["shape"]['h']
    project_root = os.path.join(base_path, project_name)

    RGB_HSI = ["RGB", "HSI"]

    BEFORE_csv = ['before_csv']
    FINAL_csv = ['csv_for_dataloader']

    splits = ["train", "val", "test"]

    formats = config["patch_formats"]

    ratio = [0.1, 0.2, 0.4, 0.8, 1.0]

    for before_csv, final_csv in zip(BEFORE_csv, FINAL_csv):
        for split in splits:
            for fmt in formats:

                n_lines = fmt['n_lines']
                patch_size = fmt['patch_size'] if fmt['patch_size'] != 0 else shape_h  # full width if 0
                stride_lines = fmt['stride_lines']
                stride_cols = fmt['stride_cols'] if fmt['stride_cols'] != 0 else shape_h
                format_name = f"w{n_lines}_h{patch_size}_sl{stride_lines}_sc{stride_cols}"

                path_before_csv_RGB = os.path.join(project_root, RGB_HSI[0], before_csv, split, format_name, split + '_' + format_name + '.csv')
                path_before_csv_HSI = os.path.join(project_root, RGB_HSI[1], before_csv, split, format_name, split + '_' + format_name + '.csv')
                # path_before_csv = os.path.join(project_root, before_csv, split, format_name, split + '_' + format_name + '.csv')
                path_final_csv_RGB = os.path.join(project_root, RGB_HSI[0], final_csv, split, format_name, split + '_' + format_name + '.csv')
                path_final_csv_HSI = os.path.join(project_root, RGB_HSI[1], final_csv, split, format_name, split + '_' + format_name + '.csv')

                columns = ['path_dat', 'path_npy', 'batch_size', 'C', 'H', 'W', 'label']
                df_path_before_csv = [path_before_csv_RGB, path_before_csv_HSI]
                df_path_final_csv = [path_final_csv_RGB, path_final_csv_HSI]

                for df_path_before, df_path_final, rgb_hsi in zip(df_path_before_csv, df_path_final_csv, RGB_HSI) :
                    df = pd.read_csv(df_path_before)

                    # df = pd.read_csv(path_before_csv)

                    expanded_rows = []

                    for _, row in df.iterrows():
                        for idx in range(int(row['batch_size'])):
                            expanded_rows.append({
                                'path_dat': row['path_dat'],
                                'path_npy': row['path_npy'],
                                'index': idx,
                                'label': row['label']
                            })
                    
                    expanded_df = pd.DataFrame(expanded_rows)
                    expanded_df.to_csv(df_path_final, index=False)

                    for p in ratio:
                        sampled_dfs = []
                        for label in expanded_df['label'].unique():
                            label_df = expanded_df[expanded_df['label'] == label]
                            n_samples = int(len(label_df) * p)
                            sampled_label_df = label_df.sample(n=n_samples, random_state=42)
                            sampled_dfs.append(sampled_label_df)
                        final_sampled_df = pd.concat(sampled_dfs)
                        p_str = str(int(p * 100))

                        if rgb_hsi == 'RGB' :
                            output_filename = os.path.join(project_root, RGB_HSI[0], final_csv, split, format_name, p_str + '%_' + split + '_' + format_name + '.csv')
                        else :
                            output_filename = os.path.join(project_root, RGB_HSI[1], final_csv, split, format_name, p_str + '%_' + split + '_' + format_name + '.csv')
                        final_sampled_df.to_csv(output_filename, index=False)

            print('CSV for Dataloader of', split, 'split is done')
# End csv_for_dataloader


# Start Patching
def data_patching(config_path: str):

    create_project_structure(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_name = config["project"]["project_name"]
    base_path = config["project"]["base_path"]
    project_root = os.path.join(base_path, project_name)

    RGB_HSI = ["RGB", "HSI"]

    subfolders = ["patches", "before_csv", "csv_for_dataloader"]
    splits = ["train", "val", "test"]
    
    formats = config["patch_formats"]

    for split in splits:
        print(split, 'hypercubes in progress ...')
        cube_dir = config['data'][split]['cube_dir']
        rgb_dir = config['data'][split]['rgb_dir']
        mask_dir = config['data'][split]['mask_dir']

        cube_files = get_sorted_files(cube_dir, '.bil.hdr')
        rgb_files = get_sorted_files(rgb_dir, '.png')
        mask_files = get_sorted_files(mask_dir, '.png')

        assert len(cube_files) == len(mask_files), f"Mismatch in number of cubes and masks in {split}"
        assert len(rgb_files) == len(mask_files), f"Mismatch in number of cubes and masks in {split}"

        for cube_file, rgb_file, mask_file in zip(cube_files, rgb_files, mask_files):
            cube_path = os.path.join(cube_dir, cube_file)
            rgb_path = os.path.join(rgb_dir, rgb_file)
            mask_path = os.path.join(mask_dir, mask_file)

            cube, _ = get_cube(cube_path)
            cube = torch.tensor(cube, dtype=torch.float32) # (H, W, C)

            rgb = read_image(rgb_path)

            mask = read_image(mask_path)[0]  # (H, W), assume 1 channel

            file_name = os.path.splitext(os.path.basename(mask_path))[0]

            for fmt in formats:
                n_lines = fmt['n_lines']
                patch_size = fmt['patch_size'] if fmt['patch_size'] != 0 else cube.shape[1]  # full width if 0
                stride_lines = fmt['stride_lines']
                stride_cols = fmt['stride_cols'] if fmt['stride_cols'] != 0 else cube.shape[1]

                format_name = f"w{n_lines}_h{patch_size}_sl{stride_lines}_sc{stride_cols}"

                path_to_patches_HSI = os.path.join(project_root, RGB_HSI[1], subfolders[0], split, format_name)
                path_to_patches_RGB = os.path.join(project_root, RGB_HSI[0], subfolders[0], split, format_name)

                path_to_befor_csv_HSI = os.path.join(project_root, RGB_HSI[1], subfolders[1], split, format_name)
                path_to_befor_csv_RGB = os.path.join(project_root, RGB_HSI[0], subfolders[1], split, format_name)

                patches_HSI, labels_HSI = extract_patches_HSI(
                    cube, mask=mask, n_lines=n_lines, patch_size=patch_size,
                    stride_lines=stride_lines, stride_cols=stride_cols,
                    full_width=(fmt['patch_size'] == 0)
                )

                patches_RGB, labels_RGB = extract_patches_RGB(
                    rgb, mask=mask, n_lines=n_lines, patch_size=patch_size,
                    stride_lines=stride_lines, stride_cols=stride_cols,
                    full_width=(fmt['patch_size'] == 0)
                )
                # print(labels)
                save_patches_batch(patches_HSI, labels_HSI, path_to_patches_HSI, file_name, path_to_befor_csv_HSI, split)
                save_patches_batch(patches_RGB, labels_RGB, path_to_patches_RGB, file_name, path_to_befor_csv_RGB, split)
            print(format_name, '||', cube_file,'/', len(cube_files), ' : done')
            
    csv_for_dataloader(config_path)
# End Patching


