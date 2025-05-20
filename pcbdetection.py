import streamlit as st
import os
from pathlib import Path
from ultralytics import YOLO
import subprocess
import torch
from PIL import Image, ImageDraw, ImageFont

# Streamlit setup
st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔍 PCB Defect Detection using YOLO + DenseNet")

# Safe paths for Streamlit Cloud
kaggle_dir = Path("/tmp/.kaggle")
base_path = Path("/tmp/pcb_dataset/pcb-defect-dataset")
yaml_path = base_path / "data.yaml"
model_dir = Path("/tmp/pcb_yolo_densenet/yolo_pcb_defects/weights")
model_path = model_dir / "best.pt"

# Step 1: Upload kaggle.json
st.header("Step 1: Upload kaggle.json")
kaggle_file = st.file_uploader("Upload your kaggle.json", type=["json"])

if kaggle_file:
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_path = kaggle_dir / "kaggle.json"
    with open(kaggle_path, "wb") as f:
        f.write(kaggle_file.read())
    os.chmod(kaggle_path, 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    st.success("✅ kaggle.json uploaded and saved")

    # Step 2: Download Dataset
    if st.button("📥 Download PCB Defect Dataset from Kaggle"):
        try:
            os.makedirs(base_path.parent, exist_ok=True)
            command = [
                "kaggle", "datasets", "download",
                "-d", "norbertelter/pcb-defect-dataset",
                "-p", str(base_path.parent), "--unzip"
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ Dataset downloaded and extracted!")

                # Step 3: Create data.yaml
                os.makedirs(base_path, exist_ok=True)
                with open(yaml_path, 'w') as f:
                    f.write(
                        f"path: {base_path.resolve()}\n"
                        f"train: train/images\n"
                        f"val: val/images\n"
                        f"test: test/images\n"
                        f"names:\n"
                        f"  0: missing_hole\n"
                        f"  1: mouse_bite\n"
                        f"  2: open_circuit\n"
                        f"  3: short\n"
                        f"  4: spur\n"
                        f"  5: spurious_copper\n"
                    )
                st.success("✅ data.yaml created!")
            else:
                st.error(f"❌ Kaggle Error: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Download Exception: {e}")

# Step 4: Train YOLOv8 Model
st.header("Step 2: Train YOLOv8 Model")
if st.button("🚀 Train Model"):
    if not yaml_path.exists():
        st.error("❌ data.yaml not found. Please download the dataset first.")
    else:
        try:
            model = YOLO("yolov8s.pt")
            model.train(
                data=str(yaml_path),
                epochs=10,
                imgsz=640,
                batch=8,
                name="yolo_pcb_defects",
                project="/tmp/pcb_yolo_densenet",
                device=0 if torch.cuda.is_available() else 'cpu'
            )
            st.success("✅ Training complete!")
        except Exception as e:
            st.error(f"❌ Training Error: {e}")

# Step 5: Upload Image & Detect Defects
st.header("Step 3: Upload Test Image for Detection")
uploaded_img = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])

if uploaded_img and model_path.exists():
    try:
        # Load image and model
        image = Image.open(uploaded_img).convert("RGB")
        model = YOLO(str(model_path))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        results = model.predict(source=image, conf=0.25)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.warning("❌ No defects detected.")
        else:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                text_size = draw.textbbox((x1, y1), label, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]

                draw.rectangle(
                    [(x1, y1 - text_height - 10), (x1 + text_width + 10, y1)],
                    fill="white"
                )
                draw.text((x1 + 5, y1 - text_height - 5), label, fill="black", font=font)
                draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=3)

            # Resize for display
            max_width = 600
            w_percent = max_width / float(image.size[0])
            h_size = int((float(image.size[1]) * float(w_percent)))
            image = image.resize((max_width, h_size), Image.Resampling.LANCZOS)
            st.image(image, caption="🔍 Detected Defects", use_column_width=False)
    except Exception as e:
        st.error(f"❌ Detection Error: {e}")
elif uploaded_img and not model_path.exists():
    st.warning("⚠️ Model not trained yet. Please train the model first.")
