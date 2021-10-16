import cv2
import numpy as np

#显示函数
def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_Corner_detection(img_path):
    # *********  预处理 ****************
    # 读取输入
    img = cv2.imread(img_path)
    #调整图片大小
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
    # 图片复制，保留原本图片
    orig = img.copy()
    #转化为灰度图
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    #高斯模糊处理灰度图，利于提取边缘
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #自适应阈值提取边缘
    edged = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    # *************  轮廓检测 ****************
    # 轮廓检测
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # 4个点的时候就拿出来
        if len(approx) == 4:
            screenCnt = approx
            #直接绘制角点
            for i in screenCnt:
                x, y = i.ravel()
                img_1 = cv2.circle(img, (x, y), 8, (255, 0, 0), -1)
            break
    #绘制轮廓，填充轮廓内部，提高角点检测的精度
    res = cv2.drawContours(img_1, [screenCnt], -1, (0, 255, 255), 3)

    show(res)


    # #******************************************另一种处理方法**********************************************
    # # *************  角点检测 ****************
    # # 角点检测，绘制角点
    # # 再次灰度
    # gray_1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # # Shi-Tomasi角点检测
    # corners = cv2.goodFeaturesToTrack(gray_1 , 4, 0.1, 10)
    # corners = np.int0(corners)
    # #获取角点位置，每轮绘制一个角点
    # for i in corners:
    #     x, y = i.ravel()
    #     img_1 = cv2.circle(res, (x, y), 8, (255, 0, 0), -1)
    # cv2.imshow("img", img_1)
    # cv2.waitKey()
    # # 正式拟合边缘，再次进行前面轮廓检测的操作，完成要求
    # gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray_1, (5, 5), 0)
    # noise_removal = cv2.bilateralFilter(blur, 9, 75, 75)
    # edged = cv2.adaptiveThreshold(noise_removal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    # contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    #
    # for c in cnts:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    #     if len(approx) == 4:
    #         screenCnt = approx
    #         break
    # res = cv2.drawContours(img_1, [screenCnt], -1, (0, 0, 255), 3)





    show(res)

    return res

# 图片相对地址
img = "train/1.jpg"
img_1 = edge_Corner_detection(img)

img = "train/2.jpg"
img_1 = edge_Corner_detection(img)
img = "train/3.jpg"
img_1 = edge_Corner_detection(img)
img = "train/4.jpg"
img_1 = edge_Corner_detection(img)
img = "train/5.jpg"
img_1 = edge_Corner_detection(img)
img = "train/6.jpg"
img_1 = edge_Corner_detection(img)
img = "train/7.jpg"
img_1 = edge_Corner_detection(img)
img = "train/8.jpg"
img_1 = edge_Corner_detection(img)
img = "train/9.jpg"
img_1 = edge_Corner_detection(img)
img = "train/10.jpg"
img_1 = edge_Corner_detection(img)
img = "train/11.jpg"
img_1 = edge_Corner_detection(img)
img = "train/12.jpg"
img_1 = edge_Corner_detection(img)
img = "train/13.jpg"
img_1 = edge_Corner_detection(img)
img = "train/14.jpg"
img_1 = edge_Corner_detection(img)
img = "train/15.jpg"
img_1 = edge_Corner_detection(img)
img = "train/16.jpg"
img_1 = edge_Corner_detection(img)
img = "train/17.jpg"
img_1 = edge_Corner_detection(img)
img = "train/18.jpg"
img_1 = edge_Corner_detection(img)

