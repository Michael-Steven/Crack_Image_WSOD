import cv2
import os
import pickle

dataset = list()
# dataset = pickle.load(open('/Users/michaelshan/Documents/BUAA/实验室项目/data_yinlie.pkl','rb'))

for root, dirs, files in os.walk('/home/syb/documents/Crack_Image_WSOD/data/cut/0/'):
    for file in files:
        # for macos
        if file == '.DS_Store':
            continue

        img_path = root + file
        im = cv2.imread(img_path)
        # print(im.shape)
        # resize image
        # newHeight = 400
        # newWidth = int(im.shape[1] * newHeight / im.shape[0])
        # im = cv2.resize(im, (newWidth, newHeight))

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
        numShowRects = 400
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
                if w > 20 * h or h > 20 * w:
                    continue
                bbox.append([x, y, w, h])
                cnt += 1
                # cv2.rectangle(imOut, (x, y), (x + w, y + h),
                #             (0, 0, 255), 1, cv2.LINE_AA)
            else:
                break

        print('Total Number of Region Proposals: {}, saved: {}'.format(len(rects), cnt))
        # # show output
        # cv2.imshow("Output", imOut)

        # # record key press
        # k = cv2.waitKey(0) & 0xFF

        # # more
        # if k == ord('m'):
        #     numShowRects += increment  # increase total number of rectangles to show by increment
        # # less
        # elif k == ord('l') and numShowRects > increment:
        #     numShowRects -= increment  # decrease total number of rectangles to show by increment
        # # quit
        # elif k == ord('q'):
        #     break
        # else:
        #     break
        new = dict({'image': file, 'label': 0, 'bbox': bbox})
        dataset.append(new)

for root, dirs, files in os.walk('/home/syb/documents/Crack_Image_WSOD/data/cut/1/'):
    for file in files:
        # for macos
        if file == '.DS_Store':
            continue

        img_path = root + file
        print(img_path)
        im = cv2.imread(img_path)
        # print(im.shape)
        # resize image
        # newHeight = 400
        # newWidth = int(im.shape[1] * newHeight / im.shape[0])
        # im = cv2.resize(im, (newWidth, newHeight))

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
        numShowRects = 400
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
                if w > 20 * h or h > 20 * w:
                    continue
                bbox.append([x, y, w, h])
                cnt += 1
                # cv2.rectangle(imOut, (x, y), (x + w, y + h),
                #             (0, 0, 255), 1, cv2.LINE_AA)
            else:
                break

        print('Total Number of Region Proposals: {}, saved: {}'.format(len(rects), cnt))
        # # show output
        # cv2.imshow("Output", imOut)

        # # record key press
        # k = cv2.waitKey(0) & 0xFF

        # # more
        # if k == ord('m'):
        #     numShowRects += increment  # increase total number of rectangles to show by increment
        # # less
        # elif k == ord('l') and numShowRects > increment:
        #     numShowRects -= increment  # decrease total number of rectangles to show by increment
        # # quit
        # elif k == ord('q'):
        #     break
        # else:
        #     break
        new = dict({'image': file, 'label': 1, 'bbox': bbox})
        dataset.append(new)


pickle.dump(dataset,open('/home/syb/documents/Crack_Image_WSOD/data/data_yinlie.pkl','wb'))