import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def Path_csvs(config_path='configs/config.yaml', RGB_HSI='HSI', Patch=100) :

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_name = config["project"]["project_name"]
    base_path = config["project"]["base_path"]
    shape_h = config["data"]["shape"]['h']
    project_root = os.path.join(base_path, project_name, RGB_HSI)

    FINAL_csv = ['csv_for_dataloader']

    splits = ["train", "val", "test"]

    formats = config["patch_formats"]

    FINAL_DIC = {'train': None,
                'val': None,
                'test': None,
                'name': None}


    for final_csv in FINAL_csv:
        for split in splits:
            for fmt in formats:
                if fmt['patch_size'] == Patch :

                    n_lines = fmt['n_lines']
                    patch_size = fmt['patch_size'] if fmt['patch_size'] != 0 else shape_h  # full width if 0
                    stride_lines = fmt['stride_lines']
                    stride_cols = fmt['stride_cols'] if fmt['stride_cols'] != 0 else shape_h
                    format_name = f"w{n_lines}_h{patch_size}_sl{stride_lines}_sc{stride_cols}"


                    path_final_csv = os.path.join(project_root, final_csv, split, format_name, split + '_' + format_name + '.csv')

                    FINAL_DIC[split] = path_final_csv
                    FINAL_DIC['name'] = format_name
    return FINAL_DIC
