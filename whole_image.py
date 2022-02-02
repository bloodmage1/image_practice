import cv2
import numpy as np
import os
from tqdm import tqdm
 
src1_imgs = os.listdir("C:/Users/user/Desktop/bl_line/bl/")
src2_imgs = os.listdir("C:/Users/user/Desktop/bl_line/bl_body/")
 
for i in tqdm(src1_imgs):
    
    src1 = cv2.imread('C:/Users/user/Desktop/bl_line/bl/' + str(i)) # 닭 전체 사진
    src2 = cv2.imread('C:/Users/user/Desktop/bl_line/bl_body/' + str(i)) # 닭 몸통 사진
    
    empty = np.zeros((src1.shape[0], src1.shape[1], 3), np.uint8) # 빈 이미지 윈도우

    result = cv2.matchTemplate(src1, src2, cv2.TM_SQDIFF_NORMED)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

    y, x = minLoc
    w, h, channel = src2.shape

    roi = src1[x:x+w, y:y+h]

    empty[x:x+w, y:y+h] = src2
    
    cv2.imwrite("C:/Users/user/Desktop/bl_line/mask/" + str(i), empty)
