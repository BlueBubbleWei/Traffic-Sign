# coding: utf-8


import cv2
import numpy as np
import time 

# 根据路径读取一张图片
image = cv2.imread("left0011.jpg") 
# 剪切有效区域
shapes=image.shape
image=image[int(1/5*shapes[1]):int(4/5*shapes[1]),int(1/12*shapes[0]):int(11/12*shapes[1]),:]
# 计算有效区域的面积
propotion=image.shape[0]*image.shape[1]
# BGR转HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
cv2.imshow('hsv',hsv)
end=image.copy()



# 绿色的范围
def getGreen(image):    
    lower_green = np.array([60, 40, 150])
    upper_green = np.array([95, 255, 255])
    return {'green':getRange(image,lower_green,upper_green)}

# 红色的范围
def getRed(image):    
    low_red = np.array([160, 80, 145])
    high_red = np.array([255, 255, 255])
    return {'red':getRange(image,low_red,high_red)}

# 绿色的范围
def getYellow(image):    
    low_yellow = np.array([15, 100, 125])
    high_yellow = np.array([30, 255, 255])
    return {'yellow':getRange(image,low_yellow,high_yellow)}

def getRange(image,low,high):
# cv2.inRange函数设阈值，去除背景部分
    mask = cv2.inRange(image, low, high)
    return mask

def deNoise(img):   
# 腐蚀    
    erode=cv2.erode(img,(100,100),iterations=4)
# 膨胀
    dilate=cv2.dilate(erode,(100,100),iterations=4)
#降噪（模糊处理用来减少瑕疵点）
    blur = cv2.blur(dilate, (5,5))
# 边缘检测    
    canny = cv2.Canny(blur, 150, 240)     
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))
# 闭操作    
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    return closed

def getPositions(img):
    ret,thresh = cv2.threshold(img,127,255,0)
    tmp,contours,hierarchy=cv2.findContours(thresh,1,2)
#     有效交通灯的比例
    if contours and cv2.contourArea(contours[0])/propotion >0.00001 and cv2.contourArea(contours[0])/propotion < 0.2:      
        cnt=contours[0]
#         最小外切圆
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.rectangle(end, (int(x-radius),int(y-radius)), (int(x+radius),int(y+radius)), (0, 255, 0), 2)
        return [int(y-radius),int(y+radius),int(x-radius),int(x+radius)]
    else:
        return 0

colors=[]
colors.append(getGreen(hsv))
colors.append(getRed(hsv))
colors.append(getYellow(hsv))
# 节省时间复杂度
start=time.clock()
areas=[]
sumResult=0
_max={'color':0}
for i in range(len(colors)):
    closed = deNoise(list(colors[i].values())[0])
    coordinate = getPositions(closed)
    if isinstance(coordinate,list):
        HSV=end[coordinate[0]:coordinate[1],coordinate[2]:coordinate[3]:]
        HSV = cv2.cvtColor(HSV, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV)
        v = np.mean(V)
        print('v',v)
        if v >list( _max.values())[0]:
            _max['color']=v
            _max[list(colors[i].keys())[0]] = _max.pop('color')
        sumResult+=v
        areas.append((list(colors[i].keys())[0],v))
    else:
        areas.append((list(colors[i].keys())[0],0))
        
# 按颜色值的大小排序

areas=sorted(areas,key=lambda x : (x[1]),reverse=True)
print(areas)
print(_max)
ended=time.clock()
print('total:',ended-start)
if int(areas[0][1]) != 0:
    print('The current color is:',areas[0][0])
else:
    print('没有交通灯')
cv2.waitKey(0)
cv2.destroyAllWindows()

