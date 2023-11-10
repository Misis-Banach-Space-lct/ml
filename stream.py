frames = []
objects = {}
people = {}
names = {0: 'animal', 1: 'balloon', 2: 'cart', 3: 'person'}
num_frames = 0

class Object():
  def __init__(self, id, cls):
    self.id = id
    self.cls = cls
    self.cnt = 0
    self.start = 0
    self.path = []
    self.conf = 0

class Object2():
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
        # if (obj[-2] + start_conf) / 2 > 0.5 or obj[-2] > 0.7:
        if (obj[-2] + start_conf) / 2 > 0.5:
          coords.append(obj[:4])
    return coords

def save_cadrs(result_after_track, model_predictor, model_cart):
    cadr = result_after_track

    for obj in cadr.boxes.data:
      if int(obj[-1]) == 3 or int(obj[-1]) == 0: continue

      id = int(obj[4])
      if not id in objects.keys():
          objects[id] = Object(id, int(obj[-1]))
          # objects[id].start = time.time()
          objects[id].start = num_frames

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
            cv2.imwrite('your_path' + '.jpg', image)
            objects[id].path.append('your_path' + '.jpg')

    cadrs = []
    for _, obj in objects.items():
        if obj.path != []:
          cadrs.append(obj)
    return cadrs
    # return objects

def check_person(res):
    for obj in res.boxes.data:
      if obj[-1] != 3: continue

      id = int(obj[4])
      x1, y1, x2, y2 = obj[:4]

      if id not in people.keys():
        cls = int(obj[-1])
        people[id] = Object2(id, cls)
        people[id].start_frame = num_frames

        people[id].first_x1 = x1
        people[id].first_y1 = y1
        people[id].first_x2 = x2
        people[id].first_y2 = y2

      people[id].frame_counts += 1
      people[id].end_frame = num_frames

      people[id].max_x = float(max(people[id].max_x, x1, x2))
      people[id].max_y = float(max(people[id].max_y, y1, y2))
      people[id].min_x = float(min(people[id].min_x, x1, x2))
      people[id].min_y = float(min(people[id].min_y, y1, y2))


    preds = {}
    boundary_zone = int(min(res[0].orig_shape[0], res[0].orig_shape[1]) / 4)
    # frequency_occurrence = 0.4
    fullness = 0.4

    for _, obj in people.items():
        criterion = [False, False, False]
        if obj.max_x - obj.min_x < boundary_zone or obj.max_y - obj.min_y < boundary_zone:
          criterion[0] = True
          # print(f'Объект {obj.id} стоял примерно в одной области')
        if (obj.end_frame - obj.start_frame) > 0 and obj.frame_counts / (obj.end_frame - obj.start_frame) > fullness:
          criterion[1] = True
          # print(f'Объект {obj.id} хорошо детектился')
        if obj.frame_counts > 300: # 300 поменять нужно на что-то другое
          criterion[2] = True
          # print(f'Объект {obj.id} был более, чем в половине видео')
        if False not in criterion:
          preds[obj.id] = obj
          
    return preds

# def Show(preds, result_after_tracking):
#     for _, obj in preds.items():
#       image = result_after_tracking[obj.start_frame].orig_img.copy()
#       cv2.rectangle(image, (int(obj.first_x1) + 1, int(obj.first_y1) + 1), (int(obj.first_x2) + 1, int(obj.first_y2) + 1), (0, 0, 255), 2)
#       # cv2.putText(image, 'StacionarnyTorgovec', (int(obj.first_x1), int(obj.first_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255 , 12), 2)
#       cv2.imwrite(obj.path, image)

VID_STRIDE = 20

with torch.no_grad():
  results = model.track(source = 'rtsp://admin:A1234567@188.170.176.190:8027/Streaming/Channels/101?transportmode=unicast&profile=Profile_1',
                        save = True, stream = True, tracker = "bytetrack.yaml", classes = [1, 2, 3], vid_stride = 1)
  for res in results:
    num_frames += 1
    print('Кадр обрабатывается')
    saved = save_cadrs(res, model_predictor, model_cart)
    print(check_person(res))
