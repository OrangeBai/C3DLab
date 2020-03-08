import config
import cv2
import os

img_path = r"F:\DataSet\Aggression_Out\test\000014.jpg"
img = cv2.imread(img_path)
img2 = cv2.resize(img, (288, 288), interpolation = cv2.INTER_CUBIC)

bicubic_test = os.path.join(config.test_path, 'bicubic.jpg')
cv2.imwrite(bicubic_test, img2)
