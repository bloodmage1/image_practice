import cv2
import numpy as np
 
src1 = cv2.imread('whole_image.jpg') # 개 전체 사진
src2 = cv2.imread('small_image.jpg') # 개 얼굴 사진
empty = np.zeros((src1.shape[0], src1.shape[1], 3), np.uint8) # 빈 이미지 윈도우

result = cv2.matchTemplate(src1, src2, cv2.TM_SQDIFF_NORMED)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

y, x = minLoc
w, h, channel = src2.shape

print(x,y,h,w)

roi = src1[x:x+w, y:y+h]

# add_img = cv2.add(empty[x:x+h, y:y+w], src2)

# print(add_img.shape)
# print(src1.shape)

empty[x:x+w, y:y+h] = src2

cv2.imshow("empty", empty)
cv2.waitKey()
cv2.destroyAllWindows()


