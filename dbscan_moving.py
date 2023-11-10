from google.colab.patches import cv2_imshow
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class Object3():
  def __init__(self, id, cls):
    self.id = id
    self.cls = cls
    self.start_frame = 0
    self.first_x1 = 0
    self.first_y1 = 0
    self.first_x2 = 0
    self.first_y2 = 0
    self.coordinates_x = []
    self.coordinates_y = []
    self.image = None

def moving_count(result_after_tracking):
  res = result_after_tracking

  objects3 = {}
  num_frame = 0

  for result in res:
    num_frame += 1
    for obj in result.boxes.data:
      if int(obj[-1]) == 3:
        id = int(obj[4])
        if id == 0: continue

        x1, y1, x2, y2 = obj[:4]

        if id not in objects3.keys():
          cls = int(obj[-1])
          objects3[id] = Object3(id, cls)
          objects3[id].start_frame = num_frame
          objects3[id].first_x1 = x1
          objects3[id].first_y1 = y1
          objects3[id].first_x2 = x2
          objects3[id].first_y2 = y2
          objects3[id].image = result.orig_img

        objects3[id].coordinates_x.append((x1 + x2) / 2)
        objects3[id].coordinates_y.append((y1 + y2) / 2)

  return objects3

objects3 = moving_count(frames2)

def otbor(objects):
    for obj in objects:
      if max(obj.coordinates_x) - min(obj.coordinates_x) < obj.image.shape[1] / 25 and  max(obj.coordinates_y) - min(obj.coordinates_y) < obj.image.shape[0] / 25:
        continue

      x_coords = np.array(obj.coordinates_x)
      y_coords = np.array(obj.coordinates_y)
      coords = np.vstack((x_coords, y_coords)).T
      db = DBSCAN(eps =  min(obj.image.shape[0], obj.image.shape[1]) / 70).fit(coords)
      labels = db.labels_
      if len(set(labels)) >= 3:
        print(obj.id)
        image = obj.image.copy()
        cv2.rectangle(image, (int(obj.first_x1) + 1, int(obj.first_y1) + 1), (int(obj.first_x2) + 1, int(obj.first_y2) + 1), (0, 0, 255), 2)
        cv2.imwrite('./moving' + str(obj.id) + '.jpg', image)

otbor(objects3.values())
