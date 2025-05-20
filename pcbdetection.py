import os
import zipfile

# Define folder structure and content
base_dir = "/mnt/data/pcb_defect_detector"
streamlit_dir = os.path.join(base_dir, ".streamlit")
os.makedirs(streamlit_dir, exist_ok=True)

# File contents
files = {
    "app.py": '''
import streamlit as st
import os
from pathlib import Path
from ultralytics import YOLO
import subprocess
import torch

st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔍 PCB Defect Detection using YOLO + DenseNet")

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
                        "\\n".join([f"  {i}: {name}" for i, name in enumerate(yaml_content['names'])])
                    )
                st.success("✅ data.yaml created successfully!")
            else:
                st.error(f"❌ Error: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Exception: {e}")

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
''',

    "requirements.txt": '''streamlit==1.32.2
ultralytics==8.0.170
torch==2.1.0
opencv-python-headless==4.8.0.74
Pillow
matplotlib
pyyaml
''',

    "packages.txt": '''git
wget
ffmpeg
kaggle
''',

    "runtime.txt": "python-3.10",

    ".streamlit/config.toml": '''[theme]
base="light"
primaryColor="#8e44ad"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f5f5f5"
textColor="#262730"
font="sans serif"
'''
}

# Write all files
for filename, content in files.items():
    path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# Create a zip file
zip_path = "/mnt/data/pcb_defect_detector.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for foldername, subfolders, filenames in os.walk(base_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            arcname = os.path.relpath(filepath, base_dir)
            zipf.write(filepath, arcname)

zip_path
