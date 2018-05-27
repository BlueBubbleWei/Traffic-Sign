import cv2
import numpy as np


image = cv2.imread("light7.jpg") # 根据路径读取一张图片
shapes=image.shape
print(shapes)
propotion=shapes[0]*shapes[1]
# print(type(origin),origin.dtype,origin.shape)
cv2.imshow('image2',image)
image = cv2.GaussianBlur(image, (7,7), 0)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR转HSV
end=image.copy()



# 绿色的范围
def getGreen(image):    
#     lower_green = np.array([35, 20, 150])
#     upper_green = np.array([70, 255, 255])
#     lower_green = np.array([16, 20, 200])
#     upper_green = np.array([36, 255, 255])
    lower_green = np.array([60, 40, 150])
    upper_green = np.array([95, 255, 255])
    return {'green':getRange(image,lower_green,upper_green)}

# 红色的范围
def getRed(image):    
#     low_red = np.array([0, 56, 90])
#     high_red = np.array([9, 203, 255])
    low_red = np.array([160, 80, 145])
    high_red = np.array([255, 255, 255])
    return {'red':getRange(image,low_red,high_red)}

# 绿色的范围
def getYellow(image):    
#     low_yellow = np.array([16, 20, 200])
#     high_yellow = np.array([36, 235, 255])
    low_yellow = np.array([15, 100, 125])
    high_yellow = np.array([30, 255, 255])
    return {'yellow':getRange(image,low_yellow,high_yellow)}

def getRange(image,low,high):
# cv2.inRange函数设阈值，去除背景部分
    mask = cv2.inRange(image, low, high)
    return cv2.bitwise_and(image, image, mask=mask)  

def deNoise(img):   
#灰度化
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
    cv2.imshow('closed',closed)
    cv2.imwrite('closed.jpg',closed)
    return closed

def getPositions(img):
    cv2.imshow('image多点',img)
    ret,thresh = cv2.threshold(img,127,255,0)
    tmp,contours,hierarchy=cv2.findContours(thresh,1,2)
    if contours and cv2.contourArea(contours[0])/propotion >0.000001 and cv2.contourArea(contours[0])/propotion < 0.001:
        print('--------')
        cnt=contours[0]    
        # 轮廓面积
        area=cv2.contourArea(cnt)
        # 轮廓周长
        perimeter = cv2.arcLength(cnt,True)
        # 轮廓近似
        epsilon=0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        k=cv2.isContourConvex(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        cv2.rectangle(end,(x,y),(x+w,y+h),(0,255,0),2)
        print('是否是闭合区域',k)
#         最小外切圆
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(end,center,radius,(0,0,255),2)
        cv2.rectangle(end, (int(x-radius),int(y-radius)), (int(x+radius),int(y+radius)), (0, 255, 0), 2)
        cv2.imshow('end',end)
        return [int(y-radius),int(y+radius),int(x-radius),int(x+radius)]
    else:
        return 0
    

colors=[]
colors.append(getGreen(hsv))
colors.append(getRed(hsv))
colors.append(getYellow(hsv))
# 节省时间复杂度

areas=[]
sumResult=0
for i in range(len(colors)):
    closed = deNoise(list(colors[i].values())[0])
    coordinate = getPositions(closed)
    if isinstance(coordinate,list):
        HSV=end[coordinate[0]:coordinate[1],coordinate[2]:coordinate[3]:]
        HSV = cv2.cvtColor(HSV, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV)
        v = np.mean(V)
        sumResult+=v
        areas.append((list(colors[i].keys())[0],v))
    else:
        areas.append((list(colors[i].keys())[0],0))
        
# 按颜色值的大小排序
areas=sorted(areas,key=lambda x : (x[1]),reverse=True)
print(areas)
if int(areas[0][1]) != 0:
    print('The current color is:',areas[0][0])
else:
    print('没有交通灯')
cv2.waitKey(0)
cv2.destroyAllWindows()


