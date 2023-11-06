import cv2

def get_frame_imgs_from_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_cnt/fps

    frame_counter = 1
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            res = model.track(frame, stream=True, conf=0.5)
            for data in res:
                clss = data.names
                cls_ids = data.boxes.numpy().cls
                tracked_ids = data.boxes.numpy().id
                confs = data.boxes.numpy().conf

                for i in range(len(cls_ids)):
                    if cls_ids[i] != 3:
                        cls_name = clss[cls_ids[i]]
                        xyxy = data.boxes.numpy().xyxy[i]
                        image = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

                        cv2.imwrite('./output/'+cls_name + f'_{int(tracked_ids[i])}_{int(confs[i]*100)}_{round(frame_counter * (duration/frame_cnt), 3)}.jpg', image)
            frame_counter += 1
        else:
            break