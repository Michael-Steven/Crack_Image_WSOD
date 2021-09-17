import cv2
import os

for root, dirs, files in os.walk('/Users/michaelshan/Documents/BUAA/实验室项目/数据集/隐裂/切过的/1/'):
    for file in files:
        # for macos
        if file == '.DS_Store':
            continue
        
        img_path = root + file
        im = cv2.imread(img_path)
        im = cv2.resize(im, (478, 478))

        img_out_path = '/Users/michaelshan/Documents/BUAA/实验室项目/数据集/隐裂/处理后/1/' + file
        cv2.imwrite(img_out_path, im)
