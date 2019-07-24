import cv2
import datetime
import os

test_path = r'F:\DataSet\Aggression_Out\Test'


def draw_rec(image, test_path, boxes):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))
    time_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f.jpg")
    file_path = os.path.join(test_path, time_name)
    cv2.imwrite(file_path, image)



test_path = r'F:\DataSet\Aggression_Out\Test'
image = cv2.imread(r'F:\DataSet\Aggression_Out\movie1_movie1__mp4\000057.jpg')
draw_rec(image, test_path, [[11, 11, 33, 99]])
