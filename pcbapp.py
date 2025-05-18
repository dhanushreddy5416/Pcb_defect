import streamlit as st
import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import zipfile
import json
import subprocess
import torch

st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔍 PCB Defect Detection using YOLO + DenseNet")

# Upload kaggle.json
st.header("Step 1: Provide Kaggle Credentials")
kaggle_file = st.file_uploader("Upload your kaggle.json", type=["json"])

base_path = Path("pcb_dataset/pcb-defect-dataset")
yaml_path = base_path / "data.yaml"

if kaggle_file:
    os.makedirs(Path.home()/".kaggle", exist_ok=True)
    kaggle_path = Path.home()/".kaggle"/"kaggle.json"
    with open(kaggle_path, "wb") as f:
        f.write(kaggle_file.read())
    os.chmod(kaggle_path, 0o600)
    st.success("✅ kaggle.json uploaded and saved")

    # Download dataset using Kaggle API
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
            else:
                st.error(f"❌ Error: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Exception: {e}")

# Model training section
st.header("Step 2: Train YOLOv8 Model")
if st.button("🚀 Train Model"):
    try:
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
