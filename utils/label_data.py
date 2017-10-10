import cv2
import os
import queue as Queue
import numpy as np
DATA_DIR = "/Volumes/DATA/train"

IM_ROWS = 5106
IM_COLS = 15106
ROI_SIZE = 960
alpha = 0.3 # 1是不透明，0是全透明

def on_mouse(event, x, y, flags, params):
    img, points = params['img'], params['points']
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        points.pop()

    temp = img.copy()
    if len(points) > 2:
        cv2.fillPoly(temp, [np.array(points)], (0, 0, 255))

    for i in range(len(points)):
        cv2.circle(temp, points[i], 1, (0, 0, 255))

    cv2.circle(temp, (x, y), 3, (0, 255, 0))
    cv2.imshow(params['name'], temp)

def label_img(img, label_name):
    c = 'x'
    if os.path.exists(label_name):
        tiny = cv2.imread(label_name, 1)
    else:
        tiny = np.zeros(img.shape)

    while c not in ['n', 'p', 'q']:
        cv2.namedWindow(label_name, cv2.WINDOW_AUTOSIZE)
        temp = img.copy()
        points = []
        cv2.setMouseCallback(label_name, on_mouse, {'img': temp, 'points': points, 'name': label_name})
        cv2.imshow(label_name, img)
        c = chr(cv2.waitKey(0))

        if c == 's':
            if len(points) > 0:
                cv2.fillPoly(img, [np.array(points)], (0, 0, 255))
                cv2.fillPoly(tiny, [np.array(points)], (255, 255, 255))
        elif c == 'd':
            if len(points) > 0:
                mask = np.zeros((img.shape[0], img.shape[1]))

                cv2.fillPoly(tiny, [np.array(points)], (0, 0, 0))
                cv2.fillPoly(mask, [np.array(points)], True, 1)
                mask = (mask == 1)
                img[mask] = origin_img[mask]
                temp[mask] = origin_img[mask]

    print(label_name)
    cv2.imwrite(label_name, tiny)
    cv2.destroyWindow(label_name)
    if c == 'n':
        return 0
    elif c == 'p':
        return 1
    elif c == 'q':
        return -1

if __name__ == '__main__':
    # i 是你所要标记的行，year是要标记的年份
    i = 1
    year = 2015
    # 按键说明
    # s 保存当前多边形
    # d 清除当前多边形内的所有标记
    # n 保存当前标记图片，并切换到下一张
    # p 保存当前标记图片，并返回上一张
    # q 退出程序

    pre = Queue.LifoQueue()

    j = 0
    origin_img = None
    while j < int(IM_COLS // ROI_SIZE):

        ss1 = '{}/{}/{}_{}_{}_.jpg'.format(DATA_DIR, year, i, j, ROI_SIZE)

        ss2 = '{}/mylabel_{}/{}_{}_{}_.jpg'.format(DATA_DIR, year, i, j, ROI_SIZE)

        src = cv2.imread(ss1, 1)
        origin_img = src.copy()
        if os.path.exists(ss2):
            tiny = cv2.imread(ss2, 0)
            src[tiny > 128] = src[tiny > 128] * (1 - alpha) + np.array([(0, 0, 255)]) * alpha

        status = label_img(src, ss2)
        if status == 0:
            pre.put(ss2)
            j += 1
        elif status == 1:
            try:
                pre.get(False)
                j -= 1
            except Queue.Empty:
                print('当前是第一张，不能回退了')
        elif status == -1:
            exit()