import cv2 as cv
import time
import numpy as np
from new_photo import new_photo
import json5
import masker


# CONFIG
try:
    with open('config.json5', 'r') as f:
        config = json5.load(f)

    IMAGES = config['IMAGES']
    root_fold = config['root_fold']
    obj_name = config['obj_name']
    format = config['format']
    light_manual = config['light_manual']
except:
    print('couldnt load json, using defaults')
    IMAGES = 12 #num images
    root_fold = "../samples/buddha3/buddha3/"
    obj_name = "buddha"
    format = ".png"
    light_manual = False


#Load input image array
image_array = []
masker.generate_mask(2)
for id in range(1, IMAGES+1):  # adjust range to your actual count
    filepath = f"{config['root_fold']}{id:03d}{format}"  # format = ".png", ".bmp", ".mat", etc.
    try:
        img = new_photo.load_image_flexible(filepath,color=False)
        image_array.append(img)
    except Exception as e:
        print(f"Error loading image {id}: {e}")


# LOADING LIGHTS FROM FILE
fs = cv.FileStorage(root_fold + "LightMatrix.yml", cv.FILE_STORAGE_READ)
fn = fs.getNode("Lights")
light_mat = fn.mat()

mask = cv.imread(root_fold + "mask" + '.png', cv.IMREAD_GRAYSCALE)



camera = new_photo(IMAGES, light_mat)

camera.process(image_array, mask=mask)
#camera.plot_color_albedo()
camera.plot_normal_map()
#camera.model_out()

