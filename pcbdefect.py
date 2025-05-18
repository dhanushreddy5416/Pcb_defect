import streamlit as st
import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import zipfile
import json
import subprocess
import torch

# ----------------------------- Streamlit UI Setup -----------------------------
st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔍 PCB Defect Detection using YOLO + DenseNet")

# ----------------------------- Paths -----------------------------
dataset_dir = Path("pcb_dataset")
base_path = dataset_dir / "pcb-defect-dataset"
yaml_path = base_path / "data.yaml"
kaggle_dataset = "norbertelter/pcb-defect-dataset"

# ----------------------------- Session State -----------------------------
if "dataset_downloaded" not in st.session_state:
    st.session_state.dataset_downloaded = False

# ----------------------------- Step 1: Upload kaggle.json -----------------------------
st.header("Step 1: Provide Kaggle Credentials")
kaggle_file = st.file_uploader("Upload your kaggle.json", type=["json"])

if kaggle_file:
    os.makedirs(Path.home() / ".kaggle", exist_ok=True)
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    with open(kaggle_path, "wb") as f:
        f.write(kaggle_file.read())
    os.chmod(kaggle_path, 0o600)
    st.success("✅ kaggle.json uploaded and saved")

    # ----------------------------- Step 2: Download Dataset -----------------------------
    if st.button("📥 Download PCB Defect Dataset from Kaggle"):
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            command = [
                "kaggle", "datasets", "download",
                "-d", kaggle_dataset,
                "-p", str(dataset_dir),
                "--unzip"
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ Dataset downloaded and extracted!")

                # Check if train and val folders exist
                if not (base_path / "train").exists() or not (base_path / "val").exists():
                    st.error("❌ 'train' or 'val' folder not found in extracted dataset.")
                else:
                    # Generate data.yaml
                    yaml_content = f"""
train: {base_path / "train"}
val: {base_path / "val"}
nc: 6
names: ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
"""
                    with open(yaml_path, "w") as f:
                        f.write(yaml_content.strip())

                    st.success("✅ data.yaml created successfully!")
                    st.session_state.dataset_downloaded = True
            else:
                st.error(f"❌ Error: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Exception during download: {e}")

# ----------------------------- Optional: Display Folder Structure -----------------------------
if st.checkbox("🔍 Show Dataset Folder Structure"):
    if base_path.exists():
        for path, dirs, files in os.walk(base_path):
            st.text(f"{path}: {len(files)} files")
    else:
        st.warning("⚠️ Dataset not found. Please download it first.")

# ----------------------------- Step 3: Train Model -----------------------------
st.header("Step 3: Train YOLOv8 Model")

if st.session_state.dataset_downloaded:
    if st.button("🚀 Train Model"):
        try:
            if not yaml_path.exists():
                st.error("❌ data.yaml not found. Cannot train.")
                st.stop()

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
else:
    st.info("ℹ️ Please upload Kaggle credentials and download the dataset before training.")
