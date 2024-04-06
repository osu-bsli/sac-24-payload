import cv2
import numpy as np

import os
from os import listdir
from os.path import isfile, join

from PIL import Image
import math

def jpeg_to_png(img_path='images/cap.jpeg', png_path=''):
    png_path=img_path.split('.')[0]+'.png'
    im = Image.open(img_path)
    im.save(png_path)
    print('saved to png',png_path)
    return png_path

def read_frame(png_path='images/cap.png'):
    frame = cv2.imread(png_path,0)
    return frame

def show_frame(frame):
    cv2.imshow('cap',frame)
    cv2.waitKey(0)

# split up a single frame into its constituent quadrants
# returns a list of 4 images ordered left to right, top to bottom.
# @param frame is a numpy array of size WxHx3, 
#       where W and H are width and height of image frame.
def get_quadrants(frame):
    assert type(frame)==np.ndarray, f"utils.py: get_quadrants(): Expected type np.array, instead got {type(frame)}"
    return [M for subFrame in np.split(frame,2, axis = 0) for M in np.split(subFrame,2, axis = 1)]
    
def split_into_quadrant_folders(parent_folder):
    
    img_file_list= [parent_folder+'/'+f for f in listdir(parent_folder) if isfile(join(parent_folder, f))]
    child_folder=parent_folder+'/camera'
    camera_nums=['1','2','3','4']
    child_folder_list=[child_folder+i for i in camera_nums]
    
    for folder in child_folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)
    print(child_folder_list)
    for imgfile in img_file_list:
        if imgfile.split('.')[1]!='png':
            print(f'converting {imgfile} to png...')
            imgfile=jpeg_to_png(imgfile)
        print(f'getting quads from {imgfile}...')
        np_im=cv2.imread(imgfile)
        quads=get_quadrants(np_im)

        for index,address,currentimg in zip(camera_nums,child_folder_list,quads):
            # print(f'IMAGE IS {currentimg}')
            addr=address+'/'+imgfile.split('/')[1].split('.')[0]+'__'+index+'.png'
            print(f"writing to {addr}")
            cv2.imwrite(addr,currentimg)
    
def delete_all_non_png(parent_folder):
    print(f'deleting all non-png files in {parent_folder}...')
    img_file_list= [parent_folder+'/'+f for f in listdir(parent_folder) if isfile(join(parent_folder, f)) and f.split('.')[1]!='png']
    for i in img_file_list:
        os.remove(i)

        
def convert_all_to_png(parent_folder):
    img_file_list= [parent_folder+'/'+f for f in listdir(parent_folder) if isfile(join(parent_folder, f))]
    for imgfile in img_file_list:
        if imgfile.split('.')[1]!='png' and imgfile.split('.')[0]+'png' not in img_file_list:
            print(f'converting {imgfile} to png...')
            imgfile=jpeg_to_png(imgfile)
    delete_all_non_png(parent_folder)

def rad_to_deg(rad_list, is_rvec = True):
    # print(f'rads are {rad_list}')
    li=[(x * 180)/math.pi for x in rad_list]
    # print(f'li is {li}')
    return li if not is_rvec else [x[0] for x in li]

import numpy as np


# rvec_a and rvec_b are 3x1 numpy arrays
def get_transformation_required(rvec_a, rvec_b):
    return rvec_a**-1 * rvec_b


def rodrigues_formula(theta, axis):
    """
    Rodrigues' rotation formula to convert a rotation vector to a rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cross_matrix = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
    rotation_matrix = (cos_theta * np.eye(3) +
                       (1 - cos_theta) * np.outer(axis, axis) +
                       sin_theta * cross_matrix)
    return rotation_matrix

def rotation_from_to(initial_rotation, final_rotation):
    """
    Calculate the rotation required to move from initial_rotation to final_rotation.
    """
    initial_matrix = rodrigues_formula(initial_rotation[0], initial_rotation[1])
    final_matrix = rodrigues_formula(final_rotation[0], final_rotation[1])
    
    # Calculate the rotation matrix from initial to final rotation
    rotation_matrix = np.dot(final_matrix, np.linalg.inv(initial_matrix))
    
    # Convert rotation matrix to axis-angle representation
    cos_theta = (np.trace(rotation_matrix) - 1) / 2
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    axis = 1 / (2 * np.sin(theta)) * np.array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                                        rotation_matrix[0, 2] - rotation_matrix[2, 0],
                                        rotation_matrix[1, 0] - rotation_matrix[0, 1]])
    return theta, axis

if __name__=='__main__':
    # li=[[[1,6],[2,6]],[[3,6],[4,6]]]
    # im = cv2.imread("image-4.png")
    # a,b,c,d=get_quadrants(im)
    # cv2.imshow('quadrant a',a)
    # cv2.imshow('quadrant b',b)
    # cv2.imshow('quadrant c',c)
    # cv2.imshow('quadrant d',d)
    # cv2.waitKey(0)
    # convert_all_to_png('calib_images')
    # split_into_quadrant_folders('calib_images')
    
    
    
    # test rodrigues formula:
    from_vec = [2.59150613, -1.58091633, 0.09634847] # 29
    to_vec = [2.7668811, 0.36025719, -0.92779701]   # 21
    result = rotation_from_to(from_vec, to_vec)
    
    print(result)
    
