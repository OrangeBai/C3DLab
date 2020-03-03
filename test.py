import cv2
from pre_processing.util import *

video_info = video_files[0]
data = generate_pickle(video_info, 'a')
for i in range(0, 100):
    img_path = os.path.join(r'F:\DataSet\Aggression_Out\mv1', str(i).zfill(6) + '.jpg')
    test_path = os.path.join(r'F:\DataSet\Aggression_Out\test', str(i).zfill(6) + '.jpg')
    img = cv2.imread(img_path)
    temp = data[0][i]
    bbox = cal_bbox(temp)
    cv2.rectangle(img, (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3])), (255, 0, 0))
    temp = data[1][i]
    bbox = cal_bbox(temp)
    cv2.rectangle(img, (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3])), (0, 255, 0))

    cv2.imwrite(test_path, img)
    print(1)