import random
import numpy as np
import torch
from ultralytics import RTDETR
ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

model = YOLO('/content/drive/MyDrive/best.pt')
model_predictor = RTDETR('/content/drive/MyDrive/best_rtdetrl.pt')
model_cart = YOLO('yolov8n.pt')

frames = []
VID_STRIDE = 5

with torch.no_grad():
  results = model.track(source = 'rtsp://admin:A1234567@188.170.176.190:8027/Streaming/Channels/101?transportmode=unicast&profile=Profile_1',
                        save = True, stream = True, tracker = "bytetrack.yaml", classes = [1, 2, 3])
  for res in results:
    frames.append(res)



names = {0: 'animal', 1: 'balloon', 2: 'cart', 3: 'person'}

class Object():
  def __init__(self, id, cls):
    self.id = id
    self.cls = cls
    self.cnt = 0
    self.start = 0
    self.path = ''
    self.conf = 0

def check_cart(model_cart, image):
    check = model_cart.predict(source = image, classes = [1, 2, 3, 5, 6, 7])
    for obj in check[0].boxes.data:
      if obj[-2] > 0.5:
        print('да, это велосипед\машина и т.д.')
        return False
    return True

def process_cadr(result_model_predictor, start_conf, image, model_cart):
    coords = []
    res = result_model_predictor[0].boxes
    for obj in res.data:
        # print(obj[-2], start_conf)
        if int(obj[-1]) == 2:
          if check_cart(model_cart, image) == False: continue
        if (obj[-2] + start_conf) / 2 > 0.6 or obj[-2] > 0.77:
          coords.append(obj[:4])
    return coords

def save_cadrs(result_after_track, model_predictor, model_cart):
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

        if objects[id].cnt < 20 or obj[-2] > objects[id].conf:
            objects[id].conf = obj[-2]
            objects[id].cnt += 1
            image = cadr.orig_img.copy()

            x1 = int(max(obj[0] - abs(obj[0] - obj[2]) / 4, 0))
            x2 = int(min(obj[2] + abs(obj[0] - obj[2]) / 4, cadr.orig_shape[1]))
            y1 = int(max(obj[1] - abs(obj[1] - obj[3]) / 4, 0))
            y2 = int(min(obj[3] + abs(obj[0] - obj[2]) / 4, cadr.orig_shape[0]))

            crop_img = image[y1 : y2, x1 : x2]
            coordinates = process_cadr(model_predictor.predict(source = crop_img, classes = [1, 2]), obj[-2], crop_img, model_cart)

            if len(coordinates) > 0:
              for coodninate in coordinates:
                  cv2.rectangle(crop_img, (int(coodninate[0]), int(coodninate[1])), (int(coodninate[2]), int(coodninate[3])), (0, 0, 255), 2)
                  cv2.putText(crop_img, names[objects[id].cls], (int(coodninate[0]), int(coodninate[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255 , 12), 2)
              image[y1 : y2, x1 : x2] = crop_img
              cv2.imwrite('./cadr3/' + str(num_cadr) + '.jpg', image)
              objects[id].path = str(num_cadr) + '.jpg'

    cadrs = []
    for _, obj in objects.items():
        if obj.path != '':
          cadrs.append(obj)
    return cadrs

saved = save_cadrs(frames, model_predictor, model_cart)


class Object():
  def __init__(self, id, cls):
    self.id = id
    self.cls = cls
    self.max_x = -1.0
    self.max_y = -1.0
    self.min_x = 38259285289.0
    self.min_y = 38259285289.0
    self.frame_counts = 0
    self.start_frame = 0
    self.end_frame = 0
    self.first_x1 = 0
    self.first_y1 = 0
    self.first_x2 = 0
    self.first_y2 = 0

def count_objects(result_after_tracking):
  res = result_after_tracking
  objects = {}
  num_frame = 0
  for result in res:
    num_frame += 1
    for obj in result.boxes.data:
      if int(obj[-1]) != 3: continue
      id = int(obj[4])
      x1, y1, x2, y2 = obj[:4]

      if id not in objects.keys():
        cls = int(obj[-1])
        objects[id] = Object(id, cls)
        objects[id].start_frame = num_frame

        objects[id].first_x1 = x1
        objects[id].first_y1 = y1
        objects[id].first_x2 = x2
        objects[id].first_y2 = y2

      objects[id].frame_counts += 1
      objects[id].end_frame = num_frame

      objects[id].max_x = float(max(objects[id].max_x, x1, x2))
      objects[id].max_y = float(max(objects[id].max_y, y1, y2))
      objects[id].min_x = float(min(objects[id].min_x, x1, x2))
      objects[id].min_y = float(min(objects[id].min_y, y1, y2))

  return objects

def select_objects(objects, result_after_tracking):
    if len(result_after_tracking) * VID_STRIDE < 2600: # меньше 2 минут
      print('Видео слишком короткое для корректного выявления для стационарных торговцев, могут быть ошибки!')
    preds = {}

    all_frames = len(result_after_tracking)
    boundary_zone = int(min(result_after_tracking[0].orig_shape[0], result_after_tracking[0].orig_shape[1]) / 4)
    frequency_occurrence = 0.4
    fullness = 0.4

    for _, obj in objects.items():
        criterion = [False, False, False]
        # criterion1 = False
        # criterion2 = False
        if obj.max_x - obj.min_x < boundary_zone or obj.max_y - obj.min_y < boundary_zone:
          criterion[0] = True
          # print(f'Объект {obj.id} стоял примерно в одной области')
        if (obj.end_frame - obj.start_frame) > 0 and obj.frame_counts / (obj.end_frame - obj.start_frame) > fullness:
          criterion[1] = True
          # if obj.frame_counts > 20 and obj.cls != 3:
          #   criterion1 = True
          # if obj.frame_counts > 30 and obj.cls == 3:
          #   criterion2 = True
          # print(f'Объект {obj.id} хорошо детектился')
        if obj.frame_counts / all_frames > frequency_occurrence:
          criterion[2] = True
          # print(f'Объект {obj.id} был более, чем в половине видео')
        if False not in criterion:
          preds[obj.id] = obj

    return preds

def Show(preds, result_after_tracking):
    x = 0
    for _, obj in preds.items():
      x += 1
      image = result_after_tracking[obj.start_frame].orig_img.copy()
      cv2.rectangle(image, (int(obj.first_x1) + 1, int(obj.first_y1) + 1), (int(obj.first_x2) + 1, int(obj.first_y2) + 1), (0, 0, 255), 2)

      names = {0: 'animal', 1: 'balloon', 2: 'cart', 3: 'person'}

      cv2.putText(image, 'StacionarnyTorgovec', (int(obj.first_x1), int(obj.first_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255 , 12), 2)
      cv2.imwrite('detect' + str(x) + '.jpg', image)

def Post_Processing(res):
    objects = count_objects(res)
    preds = select_objects(objects, res)
    print(preds.keys(), ': Возможные торговцы')
    Show(preds, res)

Post_Processing(frames)