# Run-Length Encode and Decode
import os
import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")
import cv2
import numpy as np
import pandas as pd
import time
import scipy.signal as signal

from PIL import Image
import skimage.io
import json
import csv
from scipy.ndimage.filters import gaussian_filter


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def test_arr_encoding():
    #img = cv2.imread("/Users/guihua.she/Documents/DSLife/object_detection/product_image_segmentation/keras-multi-label/plot.png")
    img=np.array([[0,0,0,0,0,],[1,0,0,0,0],[1,1,0,0,0,],[1,0,0,0,0]], dtype=np.uint8)
    print(img)
    print("_____________")
    seg=rle_encode(img)
    print(seg)
    print("_____________")
    img2=rle_decode(seg,img.shape)
    print(img2)
    print(img == img2)


def test_im_encoding():
    imf="/Users/guihua.she/Documents/DSLife/object_detection/product_image_segmentation/keras-multi-label/plot.png"
    img = Image.open(imf)
    img = img.convert("L")
    mval = np.mean(img)
    #WHITE, BLACK = 255, 0
    #img = img.point(lambda x: WHITE if x > mval else BLACK)
    img = img.point(lambda x: 1 if x > mval else 0)
    img.save("/Users/guihua.she/Documents/DSLife/object_detection/product_image_segmentation/keras-multi-label/plot33.png")
    print(img.size)

    seg=rle_encode(np.array(img))
    w,h = img.size
    img2=rle_decode(seg,(h,w))
    img3=Image.fromarray(img2)
    if np.array(img).all() == img2.all():
        print("match")
    else:
        print("not match")
    print(img3.size)
    img3.save("/Users/guihua.she/Documents/DSLife/object_detection/product_image_segmentation/keras-multi-label/plot55.png")



def im_encoding(im_path):
    img = Image.open(im_path)
    img = img.convert("L")
    mval = np.mean(img)
    w,h = img.size
    img = np.array(img)
    #center = img[h/2-8: h/2+8,w/2-8:w/2+8,0:3]
    center = img[int(h/2-8): int(h/2+8),int(w/2-8):int(w/2+8)]
    center_mean= np.mean(center)
    if center_mean > mval:
        arr = np.where(img > mval, 1, 0)
    else:
        arr = np.where(img > mval, 0, 1)
    seg=rle_encode(arr)
    return seg, h, w


#####################3

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_string(mask_image):
    runs = rle_encode(mask_image)
    return ' '.join(str(x) for x in runs)


def im_encoding2(im_path):
    img = Image.open(im_path)
    img = img.convert("L")
    mval = np.mean(img)
    w,h = img.size
    img = np.array(img)
    arr = np.where(img > mval, 1, 0)
    arr = np.array(arr, dtype=np.uint8)
    arr = np.transpose(arr)
    """
    #center = img[h/2-8: h/2+8,w/2-8:w/2+8,0:3]
    center = img[int(h/2-8): int(h/2+8),int(w/2-8):int(w/2+8)]
    center_mean= np.mean(center)
    if center_mean > mval:
        arr = np.where(img > mval, 1, 0)
    else:
        arr = np.where(img > mval, 0, 1)
    img = np.zeros(w,h, dtype=np.uint8)
    """

    seg=rle_string(arr.flatten())
    return seg, h, w




def im_encoding3(im_path):
    img = Image.open(im_path)
    img = img.convert("L")
    mval = np.mean(img)
    w,h = img.size

    img = np.array(img)
    center = img[int(h/2-h/4): int(h/2+h/4),int(w/2-w/4):int(w/2+w/4)]
    center_mean= np.mean(center)
    img = np.array(img)
    arr = None
    if center_mean < mval:
        arr = np.where(img > mval, 0,1)
    else:
        arr = np.where(img > mval, 1, 0)
    arr = np.array(arr, dtype=np.uint8)
    arr = np.transpose(arr)

    seg=rle_string(arr.flatten())
    return seg, h, w




def im_encoding4(im_path):
    img = Image.open(im_path)
    img = img.convert("L")
    w,h = img.size
    img = np.array(img)
    img = gaussian_filter(img, sigma=7)
    img = signal.medfilt2d(img, kernel_size=9)
    mval = np.mean(img)

    center = img[int(h/2-h/4): int(h/2+h/4),int(w/2-w/4):int(w/2+w/4)]
    center_mean= np.mean(center)
    img = np.array(img)
    arr = None
    if center_mean < mval:
        arr = np.where(img > mval, 0,1)
    else:
        arr = np.where(img > mval, 1, 0)
    arr = np.array(arr, dtype=np.uint8)
    arr = np.transpose(arr)
    arr = signal.medfilt2d(arr, kernel_size=5)
    #设置卷积核
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(arr, kernel,iterations=5)

    #图像膨胀处理
    erosion = cv2.dilate(erosion, kernel, iterations=5)
    seg=rle_string(erosion.flatten())
    #seg=rle_string(arr.flatten())
    return seg, h, w



#dataset="/Users/guihua.she/Documents/DSLife/object_detection/fashion-data/images/"
dataset="/Users/guihua.she/Downloads/fashion-data/images/"

label_f="/Users/guihua.she/Documents/DSLife/object_detection/fashion-data/labels.json"
train_f="/Users/guihua.she/Documents/DSLife/object_detection/fashion-data/train.txt"

labels = None
#with open(label_f) as f:
f = open(label_f)
labels = json.load(f)
f.close()
print(labels)

rle_trainset = dataset+"/rle" #img path(as id), rle encoding, EncodedPixels, CategoryId,Height,Width, CategoryName

outf = open("rle_trainset.csv","w+")
out = csv.writer(outf)
f = open(train_f)
counter = 0
cnt = []
for line in f:
    st = line.split("/")
    cat_name=labels[st[0]]
    cat_id =st[0]
    img_id = line.strip()+".jpg"
    img_path = dataset + img_id
    if False == os.path.exists(img_path):
        continue

    counter += 1
    if counter%20 == 0 :
        out.writerows(cnt)
        cnt = []
        print(counter)
    rle, h, w = im_encoding4(img_path)
    #cnt += img_id +"|"+ rle  +"|"+ cat_id +"|"+str(h)+"|"+ str(w) +"|"+ cat_name+"\n"
    cnt.append([img_id, rle, cat_id ,h , w, cat_name])

out.writerows(cnt)
f.close()
outf.close()

