import streamlit as st
import os
from pathlib import Path
from ultralytics import YOLO
import subprocess
import torch

# Set Streamlit page config
st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔍 PCB Defect Detection using YOLO + DenseNet")

# Step 1: Upload kaggle.json
st.header("Step 1: Provide Kaggle Credentials")
kaggle_file = st.file_uploader("Upload your kaggle.json", type=["json"])

# Use /tmp directory to avoid permission issues
kaggle_dir = Path("/tmp/.kaggle")
base_path = Path("pcb_dataset/pcb-defect-dataset")
yaml_path = base_path / "data.yaml"

if kaggle_file:
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_path = kaggle_dir / "kaggle.json"
    with open(kaggle_path, "wb") as f:
        f.write(kaggle_file.read())
    os.chmod(kaggle_path, 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    st.success("✅ kaggle.json uploaded and saved")

    # Step 2: Download dataset using Kaggle API
    if st.button("📥 Download PCB Defect Dataset from Kaggle"):
        try:
            os.makedirs("pcb_dataset", exist_ok=True)
            command = [
                "kaggle", "datasets", "download",
                "-d", "norbertelter/pcb-defect-dataset",
                "-p", "pcb_dataset", "--unzip"
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ Dataset downloaded and extracted!")

                # Create data.yaml
                os.makedirs(base_path, exist_ok=True)
                yaml_content = {
                    'path': str(base_path.resolve()),
                    'train': 'train/images',
                    'val': 'val/images',
                    'test': 'test/images',
                    'names': [
                        'missing_hole',
                        'mouse_bite',
                        'open_circuit',
                        'short',
                        'spur',
                        'spurious_copper'
                    ]
                }
                with open(yaml_path, 'w') as f:
                    f.write(
                        f"path: {yaml_content['path']}\n"
                        f"train: {yaml_content['train']}\n"
                        f"val: {yaml_content['val']}\n"
                        f"test: {yaml_content['test']}\n"
                        f"names:\n" +
                        "\n".join([f"  {i}: {name}" for i, name in enumerate(yaml_content['names'])])
                    )
                st.success("✅ data.yaml created successfully!")
            else:
                st.error(f"❌ Error: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Exception: {e}")

# Step 3: Train YOLOv8 model
st.header("Step 2: Train YOLOv8 Model")
if st.button("🚀 Train Model"):
    try:
        if not yaml_path.exists():
            st.error("❌ data.yaml not found. Please download dataset first.")
        else:
            model = YOLO("yolov8s.pt")
            result = model.train(
                data=str(yaml_path),
                epochs=10,
                imgsz=640,
                batch=8,
                name="yolo_pcb_defects",
                project="pcb_yolo_densenet",
                device=0 if torch.cuda.is_available() else 'cpu'
            )
            st.success("✅ Training complete!")
    except Exception as e:
        st.error(f"❌ Error during training: {e}")
