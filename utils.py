import cv2
import numpy as np

import os
from os import listdir
from os.path import isfile, join
from scipy.spatial.transform import Rotation as R

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

def rad_to_deg(rad_list, is_rvec = False):
    # print(f'rads are {rad_list}')
    li=[(x * 180)/math.pi for x in rad_list]
    # print(f'li is {li}')
    return li if not is_rvec else [(x[0] * 180)/math.pi for x in li]

import numpy as np

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

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([
            [1,         0,                  0                   ],
            [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
            [0,         math.sin(theta[0]), math.cos(theta[0])  ]
        ])
    R_y = np.array([
            [math.cos(theta[1]),    0,      math.sin(theta[1])  ],
            [0,                     1,      0                   ],
            [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
        ])
    R_z = np.array([
            [math.cos(theta[2]),    -math.sin(theta[2]),    0],
            [math.sin(theta[2]),    math.cos(theta[2]),     0],
            [0,                     0,                      1]
        ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def get_rotation_difference(mat1, mat2):
    """ Given 2 rotation matrices, 
    find and return the rotation matrix angle difference between them.
    returns matrix diff such that (mat1 @ diff) = mat2.
    """
    return mat1.T @ mat2

def get_euler_angle_to_align_vecs(vec1, vec2):
    """get euler angle to align two unit orientation vectors"""
    mat = get_rotation_difference(vec1, vec2)
    euler_angle_rot = rotationMatrixToEulerAngles(mat)
    deg_angle_to_rotate = rad_to_deg(euler_angle_rot)
    return deg_angle_to_rotate

# returns product of all matrices in a list
def calculate_net_rotation(weights_on_path):
    eye = np.identity(3)
    for w in weights_on_path:
        eye = eye @ w
    return eye

# draw function
# from https://stackoverflow.com/questions/22785849
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items



def main():
    # li=[[[1,6],[2,6]],[[3,6],[4,6]]]
    # im = cv2.imread("image-4.png")
    # a,b,c,d=get_quadrants(im)
    # cv2.imshow('quadrant a',a)
    # cv2.imshow('quadrant b',b)
    # cv2.imshow('quadrant c',c)
    # cv2.imshow('quadrant d',d)
    # cv2.waitKey(0)
    # convert_all_to_png('calib')
    # split_into_quadrant_folders('calib')
    
    
    
    # test rodrigues formula:
    from_vec = [2.59150613, -1.58091633, 0.09634847] # 29
    to_vec = [2.7668811, 0.36025719, -0.92779701]   # 21
    result = rotation_from_to(from_vec, to_vec)
    
    print(result)
    
    
if __name__ == '__main__':
    main()