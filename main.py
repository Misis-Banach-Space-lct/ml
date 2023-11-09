import random
import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics import YOLO
import cv2

from src.detect_stationary import save_cadrs
from src.detect_human_stationary import post_processing

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

names = {0: "animal", 1: "balloon", 2: "cart", 3: "person"}


def process(video_path: str):
    model = YOLO("weights/yolov8l.pt")
    model_predictor = RTDETR("weights/rtdetrl.pt")
    model_cart = YOLO("weights/yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_cnt / fps

    vid_stride = 5 # чет придумать как связать с duration

    frames = []
    with torch.no_grad():
        results = model.track(
            source=video_path,
            save=True,
            stream=True,
            tracker="bytetrack.yaml",
            classes=[1, 2, 3],
            vid_stride=vid_stride,
        )
        for res in results:
            frames.append(res)

    saved = save_cadrs(frames, model_predictor, model_cart, fps, vid_stride)

    """
    if len(saved) > 0:
        for save in saved:
            print(f"TimeCode - {save.timestamp}")
            print(f"TimeCodeML - {save.timestampML}")
            print(f"FileName - {save.path}")
            print(f"DetectedClassId - {save.cls}")
    """

    human_saved = post_processing(frames, fps, vid_stride)

    """
    if len(human_saved) > 0:
        for key in human_saved:
            print(f"TimeCode - {human_saved[key].timestamp}")
            print(f"TimeCodeML - {human_saved[key].timestampML}")
            print(f"FileName - {human_saved[key].path}")
            print(f"DetectedClassId - {human_saved[key].cls}")
    """

if __name__ == "__main__":
    process("videos/flowers.mp4")
