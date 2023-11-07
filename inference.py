import cv2

def get_frame_imgs_from_video(model, video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_cnt/fps

    about_frames = []
    saves = []

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
                        image = cv2.rectangle(frame,
                                              (int(xyxy[0]), int(xyxy[1])),
                                              (int(xyxy[2]), int(xyxy[3])),
                                              (0, 0, 255), 2)

                        frame_time = round(frame_counter * (duration/frame_cnt), 3)
                        tracked_id = int(tracked_ids[i])
                        if (tracked_id not in saves):
                            cv2.imwrite('./output/'+cls_name +
                                        f'_{tracked_id}_{int(confs[i]*100)}_{frame_time}.jpg',
                                        image)
                            saves.append(tracked_id)
                        about_frames.append({
                            "name": cls_name,
                            "tracked_id": tracked_ids[i],
                            "conf": confs[i],
                            "frame_time": frame_time
                        })
            frame_counter += 1
        else:
            return about_frames

