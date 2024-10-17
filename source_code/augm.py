# Import stuff
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt

import albumentations as A
import random

# import pywt



def graytorgb(im_t):
    clahe1 = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(5, 5))
    clahe2 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(10, 10))
    clahe3 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(15, 15))
    c1 = clahe1.apply(im_t.astype(np.uint8))
    c2 = clahe2.apply(im_t.astype(np.uint8))
    c3 = clahe3.apply(im_t.astype(np.uint8))
    return c1, c2, c3

def sobel_edge_detector(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return (grad * 255 / grad.max()).astype(np.uint8)

def elastic_transform(im, im_mask, alpha, sigma, alpha_affine, random_state=None):
    ### from https://github.com/charlychiu/U-Net/blob/master/elastic_transform.py
    ### from https://www.kaggle.com/code/ori226/data-augmentation-with-elastic-deformations

    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    image = np.concatenate((im[...,None], im_mask[...,None]), axis=2)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    im_merge_t = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    im_t = im_merge_t[...,0]
    im_mask_t = im_merge_t[...,1]

    angle = [-10, 10]
    # angle[ix]
    ix = random.randint(0, (len(angle)-1))
    transform = A.Compose([
        A.HorizontalFlip(p=0.75),
        A.Rotate(limit=(-20, 20), border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        
        
        # A.VerticalFlip(p=0.75),
        # A.RandomBrightnessContrast(p=0.2),
        # A.RandomCrop(width=256, height=256),
        # A.CLAHE(clip_limit=1.0, tile_grid_size=(5,5), p=1.0),

    ])
    # transform = A.Affine(rotate=[5, 5], p=1, mode=cv2.BORDER_CONSTANT, fit_output=True)
    transformed = transform(image=im_t, mask=im_mask_t)
    im_t = transformed["image"]
    im_mask_t = transformed["mask"]
    
    clahe1 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(5, 5))
    ## clahe1 = cv2.createCLAHE(clipLimit=0.3, tileGridSize=(5, 5))
    im_t = clahe1.apply(im_t.astype(np.uint8))

    # im_mask_t[im_mask_t>0] = 255

    return im_t, im_mask_t



def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

if __name__ == '__main__':

# sourcecode/src/vx/lungseg/ss/mks/P90_IMG0109.png
    # Load images
    image = "P57_IMG0195.png"
    # image = "P90_IMG0109.png"
    
    im = cv2.imread("./ss/img/"+image,-1)
    im_mask = cv2.imread("./ss/mks/"+image,-1)

    # # # Draw grid lines
    draw_grid(im, 20)
    draw_grid(im_mask, 20)

    
    print("im_merge.shape[1]", im.shape[1])
    # Apply transformation on image
    im_t, im_mask_t = elastic_transform(im, im_mask,
                        im.shape[1]*1.0,
                        im.shape[1]*0.038,
                        im.shape[1]*0.0,
                        # random_state=np.random.RandomState(42)
                        )
    # im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
    print("im_t.shape, im_mask_t.shape", im_t.shape, im_mask_t.shape)

    cv2.imwrite('./ss/pp/'+image+'_p.png', im_t)
    cv2.imwrite('./ss/pp/'+image+'_m.png', im_mask_t)


    # c1, c2, c3 = graytorgb(im_t)
    # cv2.imwrite('./ss/pp/P57_IMG0195_p_clahe1.png', c1)
    # cv2.imwrite('./ss/pp/P57_IMG0195_p_clahe2.png', c2)
    # cv2.imwrite('./ss/pp/P57_IMG0195_p_clahe3.png', c3)
    # cv2.imwrite('./ss/pp/P57_IMG0195_p_clahe_c.png', cv2.cvtColor(cv2.merge([c1, c3, c2]), cv2.COLOR_RGB2BGR))


    # im_s = sobel_edge_detector(im_t)
    # cv2.imwrite('./ss/pp/P57_IMG0195_p_clahe_sobel.png', im_s)























    # im_his = cv2.equalizeHist(im_t)
    # # im_his = np.hstack((im_t,equ)) #stacking images side-by-side
    # cv2.imwrite('./ss/pp/P57_IMG0195_p_his.png', im_his)

    # # # Display result
    # # plt.figure(figsize = (16,14))
    # # plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')



    # # aug = A.ElasticTransform(alpha=0.1, sigma=120.0* 0.05)
    # aug = A.GridElasticDeform(num_grid_xy=(5, 5), magnitude=1, p=0.5)

    # # aug = A.ElasticTransform(alpha=0.01, border_mode=15)

    # random.seed(7)
    # augmented = aug(image=im, mask=im_mask)

    # image_elastic = augmented['image']
    # mask_elastic = augmented['mask']


    # cv2.imwrite('./ss/pp/P57_IMG0195_p2.png', image_elastic)
    # cv2.imwrite('./ss/pp/P57_IMG0195_m2.png', mask_elastic)



    # # A, (cH, cV, cD) = pywt.dwt2(im, 'bior1.3')
    # # cv2.imwrite('./ss/pp/P57_IMG0195_p_w_A.png', A)
    # # cv2.imwrite('./ss/pp/P57_IMG0195_p_w_cH.png', cH)

