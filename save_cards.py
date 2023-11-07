model = YOLO('/content/drive/MyDrive/best.pt')

import torch
frames = []

with torch.no_grad():
  results = model.track(source='/content/drive/MyDrive/Копия 210.VC9.8.23 Набережная (Пароход) 2023-09-15 18-00-00_000+0300 [5m0s] (1).mp4',
                        save=True, stream=True, tracker = "bytetrack.yaml", vid_stride = 5)
  for res in results:
    frames.append(res)

import cv2

class Object():
  def __init__(self, id, cls):
    self.id = id
    self.cls = cls
    self.cnt = 0
    self.start = 0


def save_cadrs(result_after_track):
    res = result_after_track
    objects = {}

    num_cadr = 0
    for cadr in res:
      num_cadr += 1
      for obj in cadr.boxes.data:
        if obj[-1] == 3:
          continue

        id = float(obj[4])
        if not id in objects.keys():
            objects[id] = Object(id, float(obj[-1]))
            objects[id].start = num_cadr

        if objects[id].cnt < 3:
            objects[id].cnt += 1
            image = cadr.orig_img.copy()

            x1 = int(max(obj[0] - 100, 0))
            x2 = int(min(obj[2] + 100, cadr.orig_shape[1]))
            y1 = int(max(obj[1] - 100, 0))
            y2 = int(min(obj[3] + 100, cadr.orig_shape[0]))

            crop_img = image[y1 : y2, x1 : x2]
            cv2.imwrite('./output/' + str(round(id, 4)) + '_' + str(int(objects[id].cnt)) + '_' + str(int(objects[id].cls)) +'.jpg', crop_img)
    return objects

saved = save_cadrs(frames)

model2 = '...'
import os
for file in os.listdir('/content/output'):
  if file[-3:] != 'jpg': continue
  model2.predict(source = '/content/output/' + file, save = True)
