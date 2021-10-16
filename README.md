# 单据检测大作业
用于票据单据的边缘检测和角点检测作业，同学们可以参考，也是网络资源的汇总和参数调节达到的效果。


单据大作业
摘  要
我使用了相关的技术方法实现了在相当场合内可以使用的发票、快递单和其他单据的边缘识别和角点检测的软件代码。在我们的实验当中使用了openCV中的相关函数，核心主要是提取灰度图、高斯模糊、自适应阈值提取边缘、轮廓检测以及Shi-Tomasi角点检测等函数。使用的IDE是Pycharm，调用了cv2，numpy库，使用语言为Python。给定一个从百度上搜集的18张单据照片，在程序运行之后输出一个框选住的画框，边缘拟合单据边缘，并且输出单据的四个角点，输出为ndarray格式。处理少量的单据性能完全可以胜任。更为可贵的是，在很多其他场景下的图片边缘角点检测都可以胜任。代码可以在github上找到，在报告下方附录也会附上一份。
关键词： openCV；角点检测；边缘检测；单据识别


cv2.cvtColor 转换函数
cv2.cvtColor(p1,p2)是颜色空间转换函数，怕是需要转换的图片，p2是需要转换成何种格式，cv2.COLOR_BGR2RGB将BGR格式转换成RGB格式，cv2.COLOR_BGR2GRAY将BGR格式转换成灰度图片。
cv2.GaussianBlur函数介绍
高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。
GaussianBlur(src,ksize,sigmaX [,dst [,sigmaY [,borderType]]]）-> dst
[1] - src输入图像；图像可以具有任意数量的通道，这些通道可以独立处理，但深度应为CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。
[2] - dst输出图像的大小和类型与src相同。
[3] - ksize高斯内核大小。 ksize.width和ksize.height可以不同，但它们都必须为正数和奇数，也可以为零，然后根据sigma计算得出。
[4] - sigmaX X方向上的高斯核标准偏差。
[5] - sigmaY Y方向上的高斯核标准差；如果sigmaY为零，则将其设置为等于sigmaX；如果两个sigmas为零，则分别从ksize.width和ksize.height计算得出；为了完全控制结果，而不管将来可能对所有这些语义
cv2.adaptiveThreshold图像二值化
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)
[1] - src需要进行二值化的一张灰度图像[2] - dst输出图像的大小和类型与src相同。
[2] - maxValue：满足条件的像素点需要设置的灰度值。（将要设置的灰度值）
[3] - adaptiveMethod：自适应阈值算法。可选ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C
[4] - thresholdType：opencv提供的二值化方法，只能THRESH_BINARY或者THRESH_BINARY_INV
[5] - blockSize：要分成的区域大小，上面的N值，一般取奇数
[6] - C：常数，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
[7] - dst：输出图像，可以忽略
我们在实际实验当中自己调节相关的参数以达到最好的识别效果。
cv2.findContours查找检测物体的轮廓
轮廓检测也是图像处理中经常用到的。OpenCV-Python接口中使用cv2.findContours()函数来查找检测物体的轮廓。
cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])
opencv2返回两个值：contours：hierarchy。opencv3会返回三个值,分别是img, countours, hierarchy
第一个参数是寻找轮廓的图像；
第二个参数表示轮廓的检索模式，有四种
[1] - cv2.RETR_EXTERNAL:表示只检测外轮廓。
[2] - cv2.RETR_LISL:检测的轮廓不建立等级关系。
[3] - cv2.RETR_CCOMP:建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
[4] - cv2.RETR_TREE:建立一个等级树结构的轮廓。
第三个参数method为轮廓的近似办法
其他关键函数
（1）cv2.arcLength(cnt， True) 计算轮廓的周长
参数说明：cnt为输入的单个轮廓值
（2）cv2.aprroxPolyDP(cnt, epsilon， True) 用于获得轮廓的近似值，使用cv2.drawCountors进行画图操作。
 		参数说明：cnt为输入的轮廓值， epsilon为阈值T，通常使用轮廓的周长作为阈值，True表示的是轮廓是闭合的。
（3）cv2.drawContours() 轮廓绘制
	cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
第一个参数是指明在哪幅图像上绘制轮廓；image为三通道才能显示轮廓
第二个参数是轮廓本身，在Python中是一个list;
第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。
另一种方案----------利用shi-Tomasi角点检测
我们也可以再次方案之上使用Shi-Tomasi角点检测，在某些环境下可以较好的将角点检测出来。
首先为了提高角点检测的准确度，我们可以考虑在刚刚方案的基础上将图像单据内部完全置空
之后再次将图片转化为灰度图，利用cv2.goodFeaturesToTrack函数进行角点检测。
可能的改进方案方案
目前我们算法二值化的方案相对来说是比较傻瓜的，对于三通道的RGB图片，如果我们在提取边缘时，可以不将图片灰度化，失去一些RGB图片中带有的边缘信息，而是综合三通道的内容进行综合分析，或许我们可以得到更好的边缘提取结果，使得角点检测的结果也更加准确。
同时本作业也未能完成要求中真正使用最小二值法进行拟合直线的要求，没有思路如何提取二值化图片中的信息，实在惭愧。
第三章	结论
在这里我们将在附件中展示所有的实验结果，并且附带所有实验数据集，以及源代码。
本实验周，我们利用openCV开源库，成功将单据边缘以及角点提取出来，并且加以标记，为之后将单据中信息更好提取提供了坚实的基础，利用自适应阈值提取边缘，利用cv2.approxPolyDP找出票据的角点，连线形成图片边缘，并且题出了提高实验效果的可能方案，并且利用Shi-Tomasi角点检测提高角点检测效果。提出了更多改进的构想。希望之后的学习当中能够利用本实验中学习到的学习方法，更好的用在科研生活中。
参考文献
[1]李正大,蒋燕.提高Shi-Tomasi角点检测精度的方法研究[J].科学中国人,2017(24):49.
[2] https://blog.csdn.net/hjxu2016/article/details/77833336
[3]https://blog.csdn.net/west_three_boy/article/details/68945760?locationNum=9&fps=1
[4] https://blog.csdn.net/LaoYuanPython/article/details/108558834
[5] https://blog.csdn.net/qq_41598072/article/details/108003184


