model = YOLO('/content/drive/MyDrive/best.pt') # YOLO от 5 ноября, 11 версия датасета

import torch
frames = []
vid_stride = 20

with torch.no_grad():
  results = model.track(source='/content/drive/MyDrive/Копия 53.VC3.2.6 Белая Набережная 2023-09-20 20-52-40_000+0300 [3m0s].mp4',
                        save=True, stream=True, tracker="bytetrack.yaml", vid_stride=vid_stride)  # Частоту регулируем
  for res in results:
    frames.append(res)

from ultralytics import RTDETR
model_predictor = RTDETR('/content/drive/MyDrive/best_rtdetrl.pt') # detr обученный на 12 версии 

import cv2

names = {0: 'animal', 1: 'balloon', 2: 'cart', 3: 'person'}

class Object():
  def __init__(self, id, cls):
    self.id = id
    self.cls = cls
    self.cnt = 0
    self.start = 0
    self.path = ''
    
def process_cadr(result_model_predictor, start_conf):
    coords = []
    res = result_model_predictor[0].boxes
    for obj in res.data:
        if int(obj[-1]) == 3 or int(obj[-1]) == 0: continue
        if (obj[-2] + start_conf) / 2 > 0.6:
          coords.append(obj[:4])
    return coords

def save_cadrs(result_after_track, model_predictor,  fps, vid_stride):
    res = result_after_track
    objects = {}

    num_cadr = 0
    for cadr in res:
      num_cadr += 1
      for obj in cadr.boxes.data:
        if int(obj[-1]) == 3 or int(obj[-1]) == 0: continue

        id = float(obj[4])
        if not id in objects.keys():
            objects[id] = Object(id, int(obj[-1]))
            objects[id].start = num_cadr

        if objects[id].cnt < 3:
            objects[id].cnt += 1
            image = cadr.orig_img.copy()

            x1 = int(max(obj[0] - 50, 0))
            x2 = int(min(obj[2] + 50, cadr.orig_shape[1]))
            y1 = int(max(obj[1] - 50, 0))
            y2 = int(min(obj[3] + 50, cadr.orig_shape[0]))

            crop_img = image[y1 : y2, x1 : x2]
            coordinates = process_cadr(model_predictor.predict(source = crop_img), obj[-2])

            if len(coordinates) > 0:
              for coodninate in coordinates:
                  cv2.rectangle(crop_img, (int(coodninate[0]), int(coodninate[1])), (int(coodninate[2]), int(coodninate[3])), (0, 0, 255), 2)
                  cv2.putText(crop_img, names[objects[id].cls], (int(coodninate[0]), int(coodninate[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255 , 12), 2)
              image[y1 : y2, x1 : x2] = crop_img
              cv2.imwrite('./cadr3/' + str(num_cadr) + f'_{num_cadr * (1/fps) * vid_stride}' + '.jpg', image)
              objects[id].path = str(num_cadr) + f'_{num_cadr * (1/fps) * vid_stride}' + '.jpg'

    cadrs = []
    for _, obj in objects.items():
        if obj.path != '':
          cadrs.append(obj)
    return cadrs

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

saved = save_cadrs(frames, model_predictor, fps, vid_stride)
