import os
import shutil
import random

# source directory (original dataset)
# Assuming the script is run from tomato-disease-detector/, and original data is in ../data/train
SOURCE_DIR = "../data/train" 
DEST_DIR = "data"

# Define the disease classes
CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", # Mapping from source name
    "Tomato__Target_Spot",                         # Mapping from source name
    "Tomato__Tomato_YellowLeaf__Curl_Virus",       # Mapping from source name
    "Tomato__Tomato_mosaic_virus",                 # Mapping from source name
    "Tomato_healthy"
]

# Mapping to clean names if source names are different
# Keys are source folder names, Values are destination folder names
CLASS_MAPPING = {
    "Tomato_Bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato_Early_blight": "Tomato_Early_blight",
    "Tomato_Late_blight": "Tomato_Late_blight",
    "Tomato_Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato_Spider_mites",
    "Tomato__Target_Spot": "Tomato_Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy": "Tomato_healthy"
}

def split_data(source_dir, dest_dir, split_ratio=(0.7, 0.15, 0.15)):
    train_ratio, val_ratio, test_ratio = split_ratio
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    for source_cls, dest_cls in CLASS_MAPPING.items():
        source_path = os.path.join(source_dir, source_cls)
        if not os.path.exists(source_path):
            print(f"Warning: Class directory '{source_cls}' not found in source.")
            continue

        files = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        num_files = len(files)
        num_train = int(num_files * train_ratio)
        num_val = int(num_files * val_ratio)
        # Remaining goes to test

        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]

        print(f"Processing {dest_cls}: Total {num_files} -> Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")

        for split, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
             dest_path = os.path.join(dest_dir, split, dest_cls)
             os.makedirs(dest_path, exist_ok=True)
             
             for file in split_files:
                 src_file = os.path.join(source_path, file)
                 dst_file = os.path.join(dest_path, file)
                 shutil.copy2(src_file, dst_file)
    
    print("Data splitting completed successfully.")

if __name__ == "__main__":
    split_data(SOURCE_DIR, DEST_DIR)
