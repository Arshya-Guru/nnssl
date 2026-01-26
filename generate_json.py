import os
import sys
import json
import glob
from pathlib import Path

# --- FIX: Add 'src' to the python path so it can find nnssl ---
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)
# -------------------------------------------------------------

try:
    # Try importing assuming the package is structured correctly in src
    from nnssl.data.raw_dataset import Collection, Dataset, Subject, Session, Image
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Debug: expected to find modules in {src_path}")
    sys.exit(1)

# --- CONFIGURATION ---
DATA_DIR = "/localscratch/apooladi/patches_16bit"  # Your absolute path
OUTPUT_FILE = "pretrain_data.json"
DATASET_NAME = "MyPatches"
MODALITY = "micr" 
# ---------------------

def create_dataset_json():
    # 1. Initialize the Collection
    collection = Collection(
        collection_index=1, 
        collection_name="PatchCollection"
    )

    # 2. Initialize the Dataset
    my_dataset = Dataset(
        dataset_index="ds001", 
        name=DATASET_NAME
    )
    
    # Get list of all .nii.gz files
    file_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))
    
    if not file_list:
        print(f"WARNING: No .nii.gz files found in {DATA_DIR}")
        return

    print(f"Found {len(file_list)} images. processing...")

    # 3. Iterate through your files and build the hierarchy
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        # Treat every patch as a unique Subject ID
        subject_id = file_name.replace(".nii.gz", "")
        
        # Create the Image object
        image_obj = Image(
            name=file_name,
            image_path=os.path.abspath(file_path),
            modality=MODALITY
        )

        # Create a "Dummy" Session
        session_obj = Session(
            session_id="ses-01",
            images=[image_obj]
        )

        # Create the Subject
        subject_obj = Subject(
            subject_id=subject_id,
            sessions={"ses-01": session_obj}
        )

        # Add subject to dataset
        my_dataset.subjects[subject_id] = subject_obj

    # 4. Add Dataset to Collection
    collection.datasets["ds001"] = my_dataset

    # 5. Export to Dictionary and Save
    json_data = collection.to_dict(relative_paths=False)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Successfully created {OUTPUT_FILE} with {len(file_list)} items.")

if __name__ == "__main__":
    create_dataset_json()
