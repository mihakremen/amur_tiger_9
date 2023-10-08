import cv2
import numpy as np
import json
import os
import random

# найти ключевые точки по idшнику
def find_keypoints(image_id, json_data):
    for idx, image in enumerate(json_data['images']):
        if image['id'] == image_id:
            needful_idx = idx
            print(json_data['annotations'][needful_idx]['keypoints'])
            return np.array(json_data['annotations'][needful_idx]['keypoints'])

# id ---> 'id.jpg'
def get_image_file(images_path, id):
    return os.path.join(images_path, str(id).zfill(6)+'.jpg')

# получить координаты x, y
def get_coords(keypoints):
    if not sum(keypoints):
        print('данные не размечены')
    coords = [x for i, x in enumerate(keypoints) if (i + 1) % 3 != 0]
    return np.array([[coords[i], coords[i+1]] for i in range(0, len(coords), 2)])

# показать только ключевые точки тигра
def show_points(image, skelet_coord):
    image = cv2.imread(image)
    for points in skelet_coord:
        cv2.circle(image, (points[0], points[1]), 2, (0, 255, 0))
    cv2.imshow('tiger_skelet', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# показать скелет тигра
def show_skelet(image, coords, skeleton):
    image = cv2.imread(image)
    for points in skeleton:
        start = tuple(coords[points[0]])
        end = tuple(coords[points[1]])
        if (sum(start) != 0) and (sum(end) != 0):
            cv2.line(image, start, end, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
    cv2.imshow('tiger_skelet', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(json_path, images_path, id):
    with open(json_path) as f:
        json_data = json.load(f)
    skeleton = json_data['categories'][0]['skeleton']
    keypoints = find_keypoints(id, json_data)
    image = get_image_file(images_path, id)
    coords = get_coords(keypoints)
    show_skelet(image, coords, skeleton)

# skeleton      <list>      Данные для отрисовки скелета тигра [['правое ухо', 'нос'], ...]
# keypoints     <list>      Данные по координатам ключевых точек тигра [x1, y1, v1], где x, y - координаты, v - признак видимости
#                                                                                                           v = 0 - объект не размечен
#                                                                                                           v = 1 -  объект размечен, не виден 
#                                                                                                           v = 2 -  объект размечен, виден

if __name__ == "__main__":
    json_path = "C:\Amur tigers\data\labels\keypoint_train.json"
    images_path = "C:\\Amur tigers\\data\\images\\train"
    id = 1691
    main(json_path, images_path, id)


    


