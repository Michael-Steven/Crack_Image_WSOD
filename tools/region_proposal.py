import cv2
import os
import pickle

dataset = list()
dataset_test = list()
# dataset = pickle.load(open('/Users/michaelshan/Documents/BUAA/实验室项目/data_yinlie.pkl','rb'))

img_cnt = 0
for root, dirs, files in os.walk('/home/syb/documents/Crack_Image_WSOD/data/cut/0/'):
    for file in files:
        # for macos
        if file == '.DS_Store':
            continue
        img_cnt += 1
        img_path = root + file
        im = cv2.imread(img_path)
        im = cv2.resize(im, (478, 478))
        ratio = 400 / 478
        # cv2.imwrite('/home/syb/documents/Crack_Image_WSOD/data/resize/0/' + file, im)
        # print(img_path)
        
        bbox = list()

        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        # set input image on which we will run segmentation
        ss.setBaseImage(im)

        method = 'f'  # f=fast, q=quality

        if method == 'f':  # fast but low recall
            ss.switchToSelectiveSearchFast()
        elif method == 'q':  # high recall but slow
            ss.switchToSelectiveSearchQuality()
        else:
            exit(1)

        # run selective search segmentation on input image
        rects = ss.process()  # f:453, q:1354

        # number of region proposals to show
        numShowRects = 500
        # increment to increase/decrease total number of reason proposals to be shown
        increment = 50

        bbox = list()

        # create a copy of original image
        imOut = im.copy()

        cnt = 0
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if i < numShowRects:
                x, y, w, h = rect  # 这种格式
                if w > 10 * h or h > 10 * w:
                    continue
                bbox.append([x * ratio, y * ratio, w * ratio, h * ratio])
                cnt += 1
                # cv2.rectangle(imOut, (x, y), (x + w, y + h),
                #             (0, 0, 255), 1, cv2.LINE_AA)
            else:
                break

        # show output
        # cv2.imshow("Output", imOut)
        if img_cnt > 462:
            break
        new = dict({'image': file, 'label': 0, 'bbox': bbox})
        if img_cnt % 10 == 0:
            dataset_test.append(new)
            print('0-{}: test    |Total Number of Region Proposals: {}, saved: {}'.format(img_cnt, len(rects), cnt))
        else:
            dataset.append(new)
            print('0-{}: train   |Total Number of Region Proposals: {}, saved: {}'.format(img_cnt, len(rects), cnt))

img_cnt = 0
for root, dirs, files in os.walk('/home/syb/documents/Crack_Image_WSOD/data/cut/1/'):
    for file in files:
        # for macos
        if file == '.DS_Store':
            continue
        img_cnt += 1
        img_path = root + file
        # print(img_path)
        im = cv2.imread(img_path)
        im = cv2.resize(im, (478, 478))
        ratio = 400 / 478
        # cv2.imwrite('/home/syb/documents/Crack_Image_WSOD/data/resize/1/' + file, im)
        # print(img_path)

        bbox = list()

        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        # set input image on which we will run segmentation
        ss.setBaseImage(im)

        method = 'f'  # f=fast, q=quality

        if method == 'f':  # fast but low recall
            ss.switchToSelectiveSearchFast()
        elif method == 'q':  # high recall but slow
            ss.switchToSelectiveSearchQuality()
        else:
            exit(1)

        # run selective search segmentation on input image
        rects = ss.process()  # f:453, q:1354

        # number of region proposals to show
        numShowRects = 500
        # increment to increase/decrease total number of reason proposals to be shown
        increment = 50

        bbox = list()

        # create a copy of original image
        imOut = im.copy()

        cnt = 0
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if i < numShowRects:
                x, y, w, h = rect  # 这种格式
                if w > 10 * h or h > 10 * w:
                    continue
                bbox.append([x * ratio, y * ratio, w * ratio, h * ratio])
                cnt += 1
                # cv2.rectangle(imOut, (x, y), (x + w, y + h),
                #             (0, 0, 255), 1, cv2.LINE_AA)
            else:
                break

        # # show output
        # cv2.imshow("Output", imOut)
        new = dict({'image': file, 'label': 1, 'bbox': bbox})
        if img_cnt % 10 == 0:
            dataset_test.append(new)
            print('1-{}: test    |Total Number of Region Proposals: {}, saved: {}'.format(img_cnt, len(rects), cnt))
        else:
            dataset.append(new)
            print('1-{}: train   |Total Number of Region Proposals: {}, saved: {}'.format(img_cnt, len(rects), cnt))


pickle.dump(dataset,open('/home/syb/documents/Crack_Image_WSOD/data/data_yinlie_new.pkl','wb'))
pickle.dump(dataset_test,open('/home/syb/documents/Crack_Image_WSOD/data/data_test_yinlie_new.pkl','wb'))