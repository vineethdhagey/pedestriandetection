import streamlit as st
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

def get_latest_predict_dir(base="runs/detect"):
    if not os.path.exists(base):
        return None
    preds = [d for d in os.listdir(base) if d.startswith("predict")]
    if not preds:
        return None
    preds.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)), reverse=True)
    return os.path.join(base, preds[0])

def run():
    st.title("YOLOv8 Predictions & Fusion Dashboard")

    # --- Show predictions ---
    result_dir = get_latest_predict_dir()
    if result_dir is None or not os.path.exists(result_dir):
        st.warning("No predictions found. Run inference first.")
        return

    images = [f for f in os.listdir(result_dir) if f.endswith(".jpg")]
    if images:
        img_file = st.selectbox("Select an image", images)
        img_path = os.path.join(result_dir, img_file)
        st.image(Image.open(img_path), caption=img_file, use_container_width=True)
    else:
        st.warning("No prediction images found in latest run.")

    # --- Show fused detections ---
    fuse_file = "runs/fuse/fused.json"
    if os.path.exists(fuse_file):
        st.subheader("Fused Detections (JSON)")
        with open(fuse_file) as f:
            fused = json.load(f)
        st.json(fused)

        # --- Visualize fused detections ---
        st.subheader("Fused Detections Visualization")
        xs, ys = [], []
        for det in fused:
            if "gp" in det:  # ground-plane coordinates
                xs.append(det["gp"][0])
                ys.append(det["gp"][1])
            elif "bbox" in det:  # fallback to bbox center
                x1, y1, x2, y2 = det["bbox"]
                xs.append((x1 + x2) / 2)
                ys.append((y1 + y2) / 2)

        if xs and ys:
            fig, ax = plt.subplots()
            ax.scatter(xs, ys, c="red", label="Pedestrians")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No coordinates available to plot.")
    else:
        st.info("No fused detections saved yet. Run `--task fuse` to generate them.")

# --- Entry point for Streamlit ---
if __name__ == "__main__":
    run()
