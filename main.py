import cv2
import numpy as np
import glob
from time import strftime, localtime

input_path = "input/*.jpg"  # jpg, png 등 확장자에 맞춰 수정 필요
bg_path = "bg/*.jpg"        # jpg, png 등 확장자에 맞춰 수정 필요

bg_img_list = list()
idx = 0

for file in glob.glob(bg_path):
    bg_img_list.append(file)

for count, file in enumerate(glob.glob(input_path)):
    
    input_img = cv2.imread(file)
    input_h, input_w, _ = input_img.shape

    try:
        bg_img = cv2.imread(bg_img_list[idx])
        idx += 1
    except IndexError:
        idx = 0
        bg_img = cv2.imread(bg_img_list[idx])
        idx += 1
    
    bg_h, bg_w, _ = bg_img.shape
    bg_img = cv2.resize(bg_img, None, fx=(input_w/bg_w), fy=(input_h/bg_h), interpolation=cv2.INTER_LINEAR)

    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    ret, mask_inv = cv2.threshold(input_gray, 240, 255, cv2.THRESH_BINARY)  # 사람 부분이 검은색
    mask = cv2.bitwise_not(mask_inv)    # 사람 부분이 흰색

    input_fg = cv2.bitwise_and(input_img, input_img, mask=mask)
    bg_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)

    dst = cv2.add(input_fg, bg_bg)

    cv2.imwrite(f"output/{strftime('%m%d-%H%M%S', localtime())}_{count}.jpg", dst)
    print(f"saved! {count}", end="\r")
    cv2.waitKey(0)