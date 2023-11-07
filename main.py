from ultralytics import YOLO, RTDETR
from inference import get_frame_imgs_from_video

if __name__ == "__main__":
    model = YOLO('weights/best_yolov8l.pt', task='detect')
    # model = RTDETR('weights/best_rtdetrl.pt')

    video_path = ["videos/flowers.mp4", "videos/video_2023-11-06_15-31-43.mp4", "videos/balloons.mp4"]
    photo_path = "photos/photo_2023-11-06_14-08-42.jpg"
    about_frames = get_frame_imgs_from_video(model, video_path[2])

    print(about_frames)

