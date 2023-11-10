import random
import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics import YOLO
import cv2

from src.detect_stationary import save_cadrs
from src.detect_human_stationary import post_processing
from src.track_stream import save_cadrs as save_cadrs_stream

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

names = {0: "animal", 1: "balloon", 2: "cart", 3: "person"}


def process(video_path: str, rtsp: bool = False):
    model = YOLO("weights/yolov8l.pt")
    model_predictor = RTDETR("weights/rtdetrl.pt")
    model_cart = YOLO("weights/yolov8n.pt")

    if rtsp:
        with torch.no_grad():
            results = model.track(
                source=video_path,
                save=True,
                stream=True,
                tracker="bytetrack.yaml",
                classes=[1, 2, 3],
            )
            num_frame = 0
            for res in results:
                num_frame += 1
                print("Кадр обрабатывается")
                saved = save_cadrs_stream(
                    res,
                    model_predictor,
                    model_cart,
                    save_path="output/frames_stream",
                    num_frame=num_frame,
                )
                """
                if len(saved) > 0:
                    for key in saved:
                        # print(f"TimeCode - {saved[key].timestamp}")
                        # print(f"TimeCodeML - {saved[key].timestampML}")
                        print(f"FileName - {saved[key].path}")
                        # print(f"DetectedClassId - {saved[key].cls}")
                """
    else:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_cnt / fps

        vid_stride = 5  # чет придумать как связать с duration

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

        saved = save_cadrs(
            # video_id=1,
            result_after_track=frames,
            model_predictor=model_predictor,
            model_cart=model_cart,
            fps=fps,
            vid_stride=vid_stride,
            save_path="output/frames",
        )

        """
        if len(saved) > 0:
            for save in saved:
                print(f"TimeCode - {save.timestamp}")
                print(f"TimeCodeML - {save.timestampML}")
                print(f"FileName - {save.path}")
                print(f"DetectedClassId - {save.cls}")
        """

        human_saved = post_processing(
            frames, fps, vid_stride, save_path="output/frames_h"
        )

        """
        if len(human_saved) > 0:
            for key in human_saved:
                print(f"TimeCode - {human_saved[key].timestamp}")
                print(f"TimeCodeML - {human_saved[key].timestampML}")
                print(f"FileName - {human_saved[key].path}")
                print(f"DetectedClassId - {human_saved[key].cls}")
        """


if __name__ == "__main__":
    process(
        video_path="rtsp://admin:A1234567@188.170.176.190:8028/Streaming/Channels/101?transportmode=unicast&profile=Profile_1",
        rtsp=True,
    )
