import os
import zipfile

# Define directory
base_dir = "/mnt/data/pcb_yolo_app"
os.makedirs(base_dir, exist_ok=True)

# Define the final working app.py (auto download from Kaggle, create data.yaml)
app_code = '''\
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
                        "\\n".join([f"  {i}: {name}" for i, name in enumerate(yaml_content['names'])])
                    )
                st.success("✅ data.yaml created successfully!")
            else:
                st.error(f"❌ Error: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Exception: {e}")

st.header("Step 2: Train YOLOv8 Model")
if st.button("🚀 Train Model"):
    if not yaml_path.exists():
        st.error("❌ data.yaml not found. Please download dataset first.")
    else:
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
'''

# Save app.py
with open(os.path.join(base_dir, "app.py"), "w") as f:
    f.write(app_code)

# requirements.txt
requirements = '''\
streamlit==1.32.2
ultralytics==8.0.170
torch==2.1.0
opencv-python-headless==4.8.0.74
Pillow
matplotlib
pyyaml
'''

with open(os.path.join(base_dir, "requirements.txt"), "w") as f:
    f.write(requirements)

# packages.txt
packages = '''\
git
wget
ffmpeg
kaggle
'''

with open(os.path.join(base_dir, "packages.txt"), "w") as f:
    f.write(packages)

# runtime.txt
with open(os.path.join(base_dir, "runtime.txt"), "w") as f:
    f.write("python-3.10")

# Create zip
zip_path = "/mnt/data/pcb_yolo_streamlit_app.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, base_dir)
            zipf.write(file_path, arcname)

zip_path
