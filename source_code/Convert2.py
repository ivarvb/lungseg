


# Lung segmentation
#$$$$$$$$$$$ https://www.kaggle.com/code/dineshsellathamby/lung-segmentation
# crea imagem good
#$$$$$$$$$$$ https://www.kaggle.com/code/kmader/extracting-lung-and-structure-rendering
#Nodules candidate
#$$$$$$$$$$$ https://www.kaggle.com/code/michalstepniewski/lung-segmentation-and-candidate-points-generation


import time
#import png
import cv2
import os
from pathlib import Path
import numpy as np
import pydicom as dcm
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.fileset import FileSet
# from pydicom.filereader import read_file

import gdcm
from os import listdir
from os.path import isfile, join
from pydicom.datadict import tag_for_keyword
from skimage import exposure
#from scipy.ndimage.filters import uniform_filter
#from scipy.ndimage.measurements import variance
from skimage import morphology
import collections
import math

import matplotlib.pyplot as plt
import warnings
from skimage.restoration import denoise_nl_means, estimate_sigma

import dicom
from pydicom import dcmread 
from pydicom.fileset import FileSet
import nrrd


from datetime import datetime
import shutil
import random


from glob import glob
import operator


from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter


from augm import elastic_transform, draw_grid
from itertools import cycle


# pathdir = "/mnt/sda6/software/frameworks/data/lugcancer/raw/tt1/2/dicomRT/Original-dcm/"

# pathds = "/mnt/sda6/software/frameworks/data/lugcancer/raw"
# /mnt/sda6/software/frameworks/data/lugcancer/raw/v2/metastase/1lesao/P31
pathds = "/mnt/sda6/software/frameworks/data/lugcancer/raw/v2/"

# pathou = "/mnt/sda6/software/frameworks/data/lugcancer/clean"
pathou = "/mnt/sda6/software/frameworks/data/lugcancer/clean/v2"


"""
ref: https://www.mathworks.com/matlabcentral/answers/453376-how-to-calculate-actual-size-of-an-object-in-a-dicom-image
"""
def areamm(areapix, pixelSpacing):
    psx = pixelSpacing[0]
    psy = pixelSpacing[1]
    area_in_mm = areapix*(psx*psy)
    return area_in_mm


def makedir(fdir):
    if not os.path.exists(fdir):
        os.mkdir(fdir)


def now():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image


"""
def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    print("slices")
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
load_scan(pathdir)
"""




def set_manual_window(hu_image, custom_center, custom_width, rescale=False):
    w_image = hu_image.copy()
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    if rescale: 
        w_image = (w_image - min_value) / (max_value - min_value)*255.0 
    return w_image


def read_pixels_hu(fi, scaling=False):
    scans = dcm.dcmread(fi)

    image = scans.pixel_array
    #image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans.RescaleIntercept
    slope = scans.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    #img = image    
    img = image + np.int16(intercept)
    
    if scaling:
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    #cv2.imwrite('out_gethi.png', img)
    return img



def get_pixels_hu():
    inputdir = "/mnt/sda6/software/frameworks/data/lugcancer/ds1/2/dicomRT/Original-dcm/"
    #inputdir = "/mnt/sda6/software/frameworks/data/lugcancer/ds1/2/segmentado/dcm/"

    file = "IMG0200.dcm"
    pathf = os.path.join(inputdir, file)
    scans = dcm.dcmread(pathf)

    image = scans.pixel_array
    #image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans.RescaleIntercept
    slope = scans.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    #img = image    
    img = image + np.int16(intercept)
    #print("image hu", image)
    
    #img = np.array(img, dtype=np.int16)
    print("image huxxx x ", img)
    #img = set_manual_window(img, -400, 1200)
    plt.imshow(set_manual_window(img, -700, 255), cmap="YlGnBu")
    plt.savefig("mygraph2.png")
    #img = (img - img.min()) / (img.max() - img.min())*255.0 
    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    cv2.imwrite('out_gethi.png', img)
    return img


def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean

    return img


# Function to take care of teh translation and windowing. 
def window_imageraw(img):
    # img[img == -2000] = 0

    img_max = np.max(img)
    img_min = np.min(img)

    img = (img - img_min) / (img_max - img_min)*255.0 
    return img

# Function to take care of teh translation and windowing. 
def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    # img[img == -2000] = 0
    
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def read_dir_dicom():
    #inputdir = "/mnt/sda6/software/frameworks/data/lugcancer/cancett12/2/dicomRT/Original-dcm/"
    #file = "IMG0000.dcm"
    #pathf = os.path.join(inputdir, file)

    pathf = "/mnt/sda6/software/frameworks/data/lugcancer/ds1/2/dicomRT/Original-dcm/IMG0200.dcm"
    
    data = dcm.dcmread(pathf)
    img = data.pixel_array.astype(float)
    window_center, window_width, intercept, slope = get_windowing(data)
    print("window_center, window_width, intercept, slope", window_center, window_width, intercept, slope)
    output = window_image(img, window_center, window_width, intercept, slope, rescale = True)
    print("output", output)

    plt.imshow(output, cmap=plt.cm.gray) 
    #plt.show()
    plt.savefig("mygraph.png")

    cv2.imwrite('out_snic.png', output)

    """
    try:
        if 'PixelData' in dcm:
            img = dcm.pixel_array.astype(float)

            shape = img.shape
            img2d = []
            #if len(shape)==3:
            #    #print("shape 3", shape)
            #    dss = dcm.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            #    rowPixelSpacing = dss.PixelSpacing[0]
            #    colPixelSpacing = dss.PixelSpacing[1]
            #    ix = int(shape[0]/2.0)
            #    #print("ix3", ix)
            #    img2d = img[ix,:,:]
            #    #shape2d = 
            if len(shape)==2:
                #print("shape 2", shape)
                #pass
                #dss = dcm.SharedFunctionalGroupsSequence.PixelMeasuresSequence[0]
                
                rowPixelSpacing = dcm.PixelSpacing[0]
                colPixelSpacing = dcm.PixelSpacing[1]
                #print("img22", file, rowPixelSpacing, colPixelSpacing)
                #print("ok1")
                img2d = img[:,:]
                #Show the image with gray color-map
                plt.imshow(img2d, cmap='gray')
                #Don't show tha axes
                plt.axis('off')
                #Add a title to the plot
                plt.title('Axial Slice')
                plt.show()

                #img2d[img2d <= -1000] = 0

                #print("img2d.shape", img2d.shape)
                #print("ok2")
                image_p = img2d
                image_p = make_lungmask(image_p)
                image_p = (np.maximum(image_p,0) / image_p.max()) * 255.0
                image_p = np.uint8(image_p)
                #print(image_p)

                #with open(os.path.join(inputdir, dcm.SOPInstanceUID+'.png'), 'wb') as png_file:
                #    w = png.Writer(img2d.shape[1], img2d.shape[0], greyscale=True)
                #    w.write(png_file, image_p)
                
                plt.savefig("mygraph.png")

                cv2.imwrite('out_snic.png', image_p)

                print("imgXX", dcm.PatientID, shape, rowPixelSpacing, colPixelSpacing, image_p.max(), image_p.min())
    except:
        print("An exception occurred")
    """


"""
@ return windowning image <- MEDICOS RADIOLOGISTAS    
lung: -600, 1500
lung: -500, 1600
lung: 800, 2000 # MEDICOS RADIOLOGISTAS    
"""
def get_image_set_window(fraw, WL, WW):
    pixels = read_pixels_hu(fraw)
    pixels = set_manual_window(pixels, WL, WW, True)#
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(7, 7))
    pixels = clahe.apply(pixels.astype(np.uint8)) #local equailzation 
    return pixels


def getimagesmask(mask, img1, img2, img3, img4):
    img1p, img2p, img3p, img4p = img1.copy(), img2.copy(), img3.copy(), img4.copy()
    indexx = np.where(mask!=255)
    img1p[indexx] = 0
    img2p[indexx] = 0
    img3p[indexx] = 0
    img4p[indexx] = 0
    return img1p, img2p, img3p, img4p

def aug_combine_slide_gt(gt1, gt2):
    w = gt1.shape[0]
    h = gt1.shape[1] 
    im = np.zeros(w,h, dtype=np.uint8)

    im[np.where(gt1==1)] = 1
    im[np.where(gt2==1)] = 1

    return img
    
def aug_combine_slide(im1, im2):
    w = im1.shape[0]
    h = im2.shape[1] 
    im = np.zeros(w,h, dtype=np.uint8)

    for x in range(w):
        for y in range(h):
            im[x, y] = (im1[x, y]+im2[x, y])/2.0

    imgr = cv2.merge([im, im, im])
    return imgr
    
def aug_combine(pth):
    slices = sorted(glob(os.path.join(pth, "*.dcm")))
    for i in range(len(slices)):
        file_name = os.path.basename(slices[i])
        slices[i] = os.path.splitext(file_name)[0]
    #     print(slices[i])
    # print("")
    # print("slicesslices", slices)

    return slices 

def readrr(ds, limi, exclude=[]):
    pathoux = os.path.join(pathou, ds)
    makedir(pathoux)
    pathoux = os.path.join(pathou, ds, now())
    makedir(pathoux)

    # ds = dataset primaria, metastase
    ######### lesoes = 1leasao 2lessoes
    # ix = id do caso clinico

    #fi = "/mnt/sda6/software/frameworks/data/lugcancer/ds1/1/Segmentado/nrrd/Segmentation-fat-label.nrrd"    
    # Read the data back from file    
    # fi = "{}/{}/{}/Segmentado/nrrd/Segmentation-fat-label.nrrd".format(pathds, ds, ix)

    # fi = "{}/{}/{}/{}/Segmentado/nrrd/Segmentation-fat-label.nrrd".format(pathds, lesoes, ds, ix)

    # data/lugcancer/raw/v2/metastase/1lesao/P31



    makedir(os.path.join(pathoux))
    makedir(os.path.join(pathoux, "images"))
    makedir(os.path.join(pathoux, "images","whole_windowing"))
    makedir(os.path.join(pathoux, "images","whole_manualwindowing"))
    makedir(os.path.join(pathoux, "images","whole_radiologists"))
    makedir(os.path.join(pathoux, "images","whole_CLAHE"))
    makedir(os.path.join(pathoux, "images","whole_raw"))
    makedir(os.path.join(pathoux, "images","whole_dcm"))
    
    makedir(os.path.join(pathoux, "images","roi_thorax_windowing"))
    makedir(os.path.join(pathoux, "images","roi_thorax_manualwindowing"))
    makedir(os.path.join(pathoux, "images","roi_thorax_radiologists"))
    makedir(os.path.join(pathoux, "images","roi_thorax_CLAHE"))

    makedir(os.path.join(pathoux, "images","roi_lung_windowing"))
    makedir(os.path.join(pathoux, "images","roi_lung_manualwindowing"))
    makedir(os.path.join(pathoux, "images","roi_lung_radiologists"))
    makedir(os.path.join(pathoux, "images","roi_lung_CLAHE"))

    makedir(os.path.join(pathoux, "images","roi_lung_ch_windowing"))
    makedir(os.path.join(pathoux, "images","roi_lung_ch_manualwindowing"))
    makedir(os.path.join(pathoux, "images","roi_lung_ch_radiologists"))
    makedir(os.path.join(pathoux, "images","roi_lung_ch_CLAHE"))

    makedir(os.path.join(pathoux, "process"))

    makedir(os.path.join(pathoux, "groundtruth"))
    makedir(os.path.join(pathoux, "groundtruth", "mask"))
    makedir(os.path.join(pathoux, "groundtruth", "contour"))
    makedir(os.path.join(pathoux, "groundtruth", "roi_thorax"))
    makedir(os.path.join(pathoux, "groundtruth", "roi_lung"))
    makedir(os.path.join(pathoux, "groundtruth", "roi_lung_ch"))

    makedir(os.path.join(pathoux, "roi"))
    makedir(os.path.join(pathoux, "roi", "thorax"))
    makedir(os.path.join(pathoux, "roi", "lung"))
    makedir(os.path.join(pathoux, "roi", "lung_ch"))

    # typereg = {"nodules":{"idx":{}, "slides":{}},  "masses":{}}
    for lesoes in [ "1lesao","2lesoes"]:
        pt = os.path.join(pathds, ds, lesoes)
        
        for ix in os.listdir(os.path.join(pt)):
            fi = os.path.join(pt, ix, "Segmentado/nrrd/Segmentation-fat-label.nrrd")
            
            makedir(os.path.join(pathoux, "images", "whole_dcm", ix))

            pathprocess = os.path.join(pathoux, "process", ix)
            makedir(pathprocess)

            if os.path.exists(fi) and not ix in exclude:

                readdata, header = nrrd.read(fi)
                
                silesn = matchs(os.path.join(pt, ix, "dicomRT", "Original-dcm"))
                
                # for each slide
                for z in range(readdata.shape[2]):
                    sli = readdata[:, :, z]
                    #if ix != 8:
                    sli = sli.T

                    seg = np.where(sli!=0)

                    binx2 = np.zeros([readdata.shape[0],readdata.shape[1]], dtype=np.uint8)
                    binx = np.zeros([readdata.shape[0],readdata.shape[1]], dtype=np.uint8)
                    binx[seg] = 255
        
                    # th, dst = cv2.threshold(binx, 0, 255, cv2.THRESH_BINARY); 
                    # dst = dst.astype(np.uint8)

                    retx, labelsx = cv2.connectedComponents(binx)                    
                    maskx = np.array(labelsx, dtype=np.uint8)

                    ## ****** begin read dcm file
                    i = silesn[z]
                    fraw = os.path.join(pt, ix, "dicomRT", "Original-dcm", i+".dcm" )
                    data = dcm.dcmread(fraw)
                    # ### begin compute pixelspacing
                    # print("PixelSpacing.PixelSpacing", data.PixelSpacing)
                    # print("datdcmf.PixelSpacing", datdcmf)
                    # datdcmfm = datdcmf.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
                    # rowPixelSpacing = datdcmfm.PixelSpacing[0]
                    # colPixelSpacing = datdcmfm.PixelSpacing[1]
                    # print("rowPixelSpacing, colPixelSpacing", rowPixelSpacing, colPixelSpacing)
                    # ### end compute pixelspacing
                    ## ****** end read dcm file


                    # for each component
                    for lbl in range(1,retx):
                        cc = np.where(maskx==lbl)
                        areai = areamm(len(cc[0]), data.PixelSpacing)
                        # 7.5*2 = 15 = 30/2 of node diameter
                        # arealimiinf = 3.14159265*((7.5)**2)
                        # print("arealimiinf, areai", len(cc[0]), arealimiinf, areai)
                        
                        if len(cc[0])>=limi:
                        #if len(cc[0])>=limi:
                            binx2[cc] = 255
                    seg = np.where(binx2==255)
                    del binx

                    if len(seg[0])>=limi:
                        # print("areai", areai)
                        ## read windowning
                        img = data.pixel_array
                        window_center, window_width, intercept, slope = get_windowing(data)
                        print("window_center, window_width, intercept, slope", window_center, window_width, intercept, slope)
                        pixelsraw = window_imageraw(img)
                        pixels_lung = window_image(img, window_center, window_width, intercept, slope, rescale = True)


                        ## read without windowning
                        #pixels = read_pixels_hu(fraw, scaling=True)


                        #************
                        # pixelsd = window_image(img, window_center, window_width, intercept, slope, rescale = False)
                        # psave = True
                        psave = False
                        imgthorax, imglung, imglung_ch, contours_thorax, contours_lung, contours_lung_ch = getlung(pixels_lung, pathprocess, ix, i, PSAVE=psave)

                        joincl = cv2.bitwise_or(imglung, binx2)
                        excl = cv2.bitwise_xor(imglung, joincl)
                        iexcl = np.where(excl>0)
                        del joincl
                        del excl
                        print("iexcl", iexcl)
                        if len(iexcl[0])==0:
                            cv2.imwrite(os.path.join(pathoux, "roi", "thorax", ix+"_"+i+".png"), imgthorax)
                            cv2.imwrite(os.path.join(pathoux, "roi", "lung", ix+"_"+i+".png"), imglung)
                            cv2.imwrite(os.path.join(pathoux, "roi", "lung_ch", ix+"_"+i+".png"), imglung_ch)
                            # exit(0)
                            #***********************
                            #***********************

                            # print("pixels", pixels)
                            # window_width = window_width-100
                            # intercept = -600
                            # pixels = window_image(img, window_center, window_width, intercept, slope, rescale = True)
                            # pixels = pixelsraw
                            
                            
                            # with hu and windowning:
                            ######### https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
                            ######### https://www.kaggle.com/code/redwankarimsony/ct-scans-dicom-files-windowing-explained
                            #****** FALTA REVISAR ESTE https://gist.github.com/lebedov/e81bd36f66ea1ab60a1ce890b07a6229
                            #****** https://www.stepwards.com/?page_id=21646
                            pixels_manual = get_image_set_window(fraw, -600, 1500)
                            pixels_radiologists = get_image_set_window(fraw, 800, 2000)
                            #***********************
                            #***********************

                            cv2.imwrite(os.path.join(pathoux, "images", "whole_windowing", ix+"_"+i+".png"), pixels_lung)
                            cv2.imwrite(os.path.join(pathoux, "images", "whole_manualwindowing", ix+"_"+i+".png"), pixels_manual)
                            cv2.imwrite(os.path.join(pathoux, "images", "whole_radiologists", ix+"_"+i+".png"), pixels_radiologists)
                            # equalized = cv2.equalizeHist(pixels.astype(np.uint8))
                            # cv2.imwrite(os.path.join(pathoux, "images", "equalizeHist", ix+"_"+i+".png"), equalized)
                            clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(7, 7))
                            pixels_lung_clahe = clahe.apply(pixels_lung.astype(np.uint8))
                            cv2.imwrite(os.path.join(pathoux, "images", "whole_CLAHE", ix+"_"+i+".png"), pixels_lung_clahe)
                            cv2.imwrite(os.path.join(pathoux, "images","whole_raw", ix+"_"+i+".png"), pixelsraw)
                            shutil.copyfile(fraw,  os.path.join(pathoux, "images", "whole_dcm", ix, ix+"_"+i+".dcm"))


                            # foi = os.path.join(pathoux, "mask", ix+"_"+i+".jpg")
                            # pixelsn = np.zeros([readdata.shape[0],readdata.shape[1]], dtype=np.uint8)
                            # pixelsn[seg] = 255
                            # cv2.imwrite(foi, pixelsn)

                            # save mask GT
                            # pixelsn = np.zeros([readdata.shape[0],readdata.shape[1]], dtype=np.uint8)
                            # pixelsn[seg] = 255
                            cv2.imwrite(os.path.join(pathoux, "groundtruth", "mask", ix+"_"+i+".png"), binx2)


                            gt_thorax = cv2.bitwise_and(imgthorax, binx2)
                            gt_lung = cv2.bitwise_and(imglung, binx2)
                            gt_lung_ch = cv2.bitwise_and(imglung_ch, binx2)
                            cv2.imwrite(os.path.join(pathoux, "groundtruth", "roi_thorax", ix+"_"+i+".png"), gt_thorax)
                            cv2.imwrite(os.path.join(pathoux, "groundtruth", "roi_lung", ix+"_"+i+".png"), gt_lung)
                            cv2.imwrite(os.path.join(pathoux, "groundtruth", "roi_lung_ch", ix+"_"+i+".png"), gt_lung_ch)
                            

                            # save contours
                            contours_gt, hierarchy = cv2.findContours(binx2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            # imgc_gt = np.stack((pixels_lung,)*3, axis=-1)
                            # imgc_gt = np.stack((pixelsraw,)*3, axis=-1)
                            imgc_gt = np.stack((pixels_manual,)*3, axis=-1)
                            
                            
                            imgc_thorax = imgc_gt.copy()
                            imgc_lung = imgc_gt.copy()
                            imgc_lung_ch = imgc_gt.copy()
                            imgc_whole = imgc_gt.copy()

                            if psave:    
                                cv2.drawContours(imgc_thorax, contours_thorax, -1, (0,0,255), 2)# lungch mask
                                cv2.imwrite(os.path.join(pathoux, "groundtruth", "contour", ix+"_"+i+"_thorax.png"), imgc_thorax)
                                cv2.drawContours(imgc_lung, contours_lung, -1, (255,0,0), 2)# lung mask
                                cv2.imwrite(os.path.join(pathoux, "groundtruth", "contour", ix+"_"+i+"_lung.png"), imgc_lung)
                                cv2.drawContours(imgc_lung_ch, [contours_lung_ch], -1, (255,0,255), 2)# lungch mask
                                cv2.imwrite(os.path.join(pathoux, "groundtruth", "contour", ix+"_"+i+"_lung_ch.png"), imgc_lung_ch)
                                cv2.drawContours(imgc_gt, contours_gt, -1, (0,255,0), 2)# gt mask
                                cv2.imwrite(os.path.join(pathoux, "groundtruth", "contour", ix+"_"+i+"_gt.png"), imgc_gt)

                            cv2.drawContours(imgc_whole, contours_thorax, -1, (0,0,255), 2)# lungch mask
                            cv2.drawContours(imgc_whole, [contours_lung_ch], -1, (255,0,255), 2)# lungch mask
                            cv2.drawContours(imgc_whole, contours_lung, -1, (255,0,0), 2)# lung mask
                            cv2.drawContours(imgc_whole, contours_gt, -1, (0,255,0), 2)# gt mask
                            cv2.imwrite(os.path.join(pathoux, "groundtruth", "contour", ix+"_"+i+"_whole.png"), imgc_whole)
                            

                            # crop mask on images processing
                            plungp, pmanualp, pradiologistsp, plungclahep = getimagesmask(imgthorax, pixels_lung, pixels_manual, pixels_radiologists, pixels_lung_clahe)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_thorax_windowing", ix+"_"+i+".png"), plungp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_thorax_manualwindowing", ix+"_"+i+".png"), pmanualp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_thorax_radiologists", ix+"_"+i+".png"), pradiologistsp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_thorax_CLAHE", ix+"_"+i+".png"), plungclahep)

                            plungp, pmanualp, pradiologistsp, plungclahep = getimagesmask(imglung, pixels_lung, pixels_manual, pixels_radiologists, pixels_lung_clahe)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_windowing", ix+"_"+i+".png"), plungp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_manualwindowing", ix+"_"+i+".png"), pmanualp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_radiologists", ix+"_"+i+".png"), pradiologistsp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_CLAHE", ix+"_"+i+".png"), plungclahep)

                            plungp, pmanualp, pradiologistsp, plungclahep = getimagesmask(imglung_ch, pixels_lung, pixels_manual, pixels_radiologists, pixels_lung_clahe)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_ch_windowing", ix+"_"+i+".png"), plungp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_ch_manualwindowing", ix+"_"+i+".png"), pmanualp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_ch_radiologists", ix+"_"+i+".png"), pradiologistsp)
                            cv2.imwrite(os.path.join(pathoux, "images", "roi_lung_ch_CLAHE", ix+"_"+i+".png"), plungclahep)

                        
                        # print(sli.shape, z, len(seg[0]), i, foi)
                        
                        del img
                        del pixels_lung

                    del data
                    del seg
                    del sli
                    del binx2
def matchs(pth):

    slices = sorted(glob(os.path.join(pth, "*.dcm")))
    for i in range(len(slices)):
        file_name = os.path.basename(slices[i])
        slices[i] = os.path.splitext(file_name)[0]
    #     print(slices[i])
    # print("")
    # print("slicesslices", slices)

    return slices 




#     dcm_name_uid = {}
#     # ss=[]
#     # for ff in os.listdir(pth):
#     #     # print("ff",ff)
#     #     if ff.endswith('.dcm'):
#     #         # print("ffxx",ff)
#     #         sss = read_file(os.path.join(pth, ff), force=True).SOPInstanceUID
#     #         # print("sss", sss)
#     #         ss.append(ff)
#     # print("ss",ss)



#     for name in [ff for ff in os.listdir(pth) if ff.endswith('.dcm') ]:
#         try:
#             # print("read_file(os.join(pth, name), force=True).SOPInstanceUID", read_file(os.path.join(pth, name), force=True).SOPInstanceUID)
#             dcm_name_uid[name] = read_file(os.path.join(pth, name), force=True).SOPInstanceUID
#         except:
#             pass
    
#     print("dcm_name_uid", dcm_name_uid)
#     rtss = read_file(os.path.join(pth, 'rtss.dcm'), force=True)
#     hits = [k for k, v in dcm_name_uid.items()
#             if v == rtss.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID]
#     print("hits", hits)
#     return hits


# def creasplit(path, n):
#     makedir(os.path.join(path, "split"))
    
#     makedir(os.path.join(path, "split", "train"))
#     makedir(os.path.join(path, "split", "train", "ground_truths"))
#     makedir(os.path.join(path, "split", "train", "ground_truths","masks"))
#     makedir(os.path.join(path, "split", "train", "render_images"))
#     makedir(os.path.join(path, "split", "train", "render_images","images"))

#     makedir(os.path.join(path, "split", "test"))
#     makedir(os.path.join(path, "split", "test", "ground_truths"))
#     makedir(os.path.join(path, "split", "test", "ground_truths","masks"))
#     makedir(os.path.join(path, "split", "test", "render_images"))
#     makedir(os.path.join(path, "split", "test", "render_images","images"))

#     piimg = os.path.join(path, "images")
#     pimask = os.path.join(path, "mask")

#     fils = [im for im in os.listdir(os.path.join(path, "images"))]

#     ids = [i for i in range(1, n+1)]
#     random.shuffle(ids)
#     cu = int(len(ids)*(10.0/100.0))
#     print("ids", ids)
#     test = ids[0:cu]
#     train = ids[cu:]


#     for dat in [("test",test),("train",train)]:
#         listn, listd = dat[0], dat[1]
#         poms = os.path.join(path, "split", listn, "ground_truths","masks")
#         poim = os.path.join(path, "split", listn, "render_images","images")

#         for iu in listd:
#             res = [i for i in fils if i.startswith(str(iu)+'_')]
#             for ims in res:
#                 shutil.copyfile(os.path.join(piimg, ims),  os.path.join(poim, ims))
#                 shutil.copyfile(os.path.join(pimask, ims), os.path.join(poms, ims))

def draw_fill_holles(img):
    imgp = img.copy()
    contours, hierarchy = cv2.findContours(imgp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        # Check if the contour has a parent
        if hierarchy[0][i][3] == -1:
            # If the contour doesn't have a parent, fill it with pixel value 255
            cv2.drawContours(imgp, [contour], -1, 255, cv2.FILLED)        
    return imgp

def get_largest_component(img_binary):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_binary, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img = np.zeros(output.shape, np.uint8)
    img[output == max_label] = 255
    return img

def saveimg(pathoux, img, aa, bb, PSAVE):
    if PSAVE:
        cv2.imwrite(os.path.join(pathoux, aa+"_"+bb+".png"), img)

def getlung(imgraw, pathoux, ix, i, PSAVE = False):
    img = np.uint8(imgraw)
    print("img.shape", img.shape)
    mxv = np.max(img)
    miv = np.min(img)
    print("mxv, miv", mxv, miv)

    ### 1) Get Binary 
    ### 1.1) add constrast
    img = cv2.medianBlur(img, 7)
    saveimg(pathoux, img, ix, i+"_1.1", PSAVE)
    # imgi = cv2.fastNlMeansDenoising(imgi, None, 16, 12, 16)

    ### 1.2) add constrast
    alpha = 1.8 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    saveimg(pathoux, img, ix, i+"_1.2", PSAVE)

    ### 1.3) thresholding
    th, img_binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    saveimg(pathoux, img_binary, ix, i+"_1.3", PSAVE)

    ### 2) Get the largets component
    ### 2.1) Get the largets component
    img_largest = get_largest_component(img_binary)
    saveimg(pathoux, img_largest, ix, i+"_2.1", PSAVE)
    ### 2.2) fill holles
    img_largest_fill_holles = draw_fill_holles(img_largest)
    contours_thorax, hierarchy = cv2.findContours(img_largest_fill_holles, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    saveimg(pathoux, img_largest_fill_holles, ix, i+"_2.2", PSAVE)


    ### 3) Get lung
    index = np.where(img_largest_fill_holles==255)

    ### 3.1) get lung by difference
    img_lung = np.zeros(img.shape, np.uint8)
    img_lung[index] = img_largest_fill_holles[index]-img_largest[index]
    # img_lung = cv2.bitwise_xor(img_largest_fill_holles, img_largest)
    saveimg(pathoux, img_lung, ix, i+"_3.1", PSAVE)
    ### 3.2) delete noises
    kernel = np.ones((5,5),np.uint8)
    img_lung = cv2.morphologyEx(img_lung, cv2.MORPH_OPEN, kernel)
    saveimg(pathoux, img_lung, ix, i+"_3.2", PSAVE)

    ### 3.2) suavizar contornos
    kernelell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # kernel = np.ones((25,25),np.uint8)
    img_lung = cv2.dilate(img_lung, kernelell, iterations=1)
    # img_lung = cv2.dilate(img_lung, kernelell, iterations=1)
    saveimg(pathoux, img_lung, ix, i+"_3.3", PSAVE)
    ### 3.4) erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_lung = cv2.erode(img_lung, kernel, iterations = 1)
    saveimg(pathoux, img_lung, ix, i+"_3.4", PSAVE)
    ## filllllllllllllllllllllllllllllllllllllllllllll falta llenar hueco
    kernelll = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    img_lung = cv2.morphologyEx(img_lung, cv2.MORPH_CLOSE, kernelll)



    # ## Allllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll
    # ### 3.3) dilate
    # img_lung_cp = img_lung.copy()
    # kernel = np.ones((25,25),np.uint8)
    # img_lung_cp = cv2.dilate(img_lung_cp, kernel, iterations=1)
    # saveimg(pathoux, img_lung_cp, ix, i+"_3.3", PSAVE)
    # ### 3.4) erode
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # img_lung_cp = cv2.erode(img_lung_cp, kernel, iterations = 1)
    # saveimg(pathoux, img_lung_cp, ix, i+"_3.4", PSAVE)
    # # #$$$$$$$$$$ OR
    # # # suavisa a morfologia
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 10))
    # # #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    # # img_lung = cv2.morphologyEx(img_lung, cv2.MORPH_OPEN, kernel)    
    # # saveimg(pathoux, img_lung, ix, i+"_3.3", PSAVE)
    # ## Allllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll


    ### 3.x) get contour
    contours_lung, hierarchy = cv2.findContours(img_lung, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    ### 4) Get convex hull lung
    ##  4.1) dilate
    kernel = np.ones((25,25),np.uint8)
    img_lung_ch = cv2.dilate(img_lung, kernel, iterations=1)
    saveimg(pathoux, img_lung_ch, ix, i+"_4.1", PSAVE)
    ##  4.2) erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    img_lung_ch = cv2.erode(img_lung_ch, kernel, iterations = 1)
    saveimg(pathoux, img_lung_ch, ix, i+"_4.2", PSAVE)

    ##  4.3) get convex hull
    contours_lung_ch, hierarchy = cv2.findContours(img_lung_ch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_lung_ch = np.vstack(contours_lung_ch)
    contours_lung_ch = cv2.convexHull(contours_lung_ch)
    
    img_lung_ch = np.zeros(img.shape, np.uint8)
    # cv2.fillpoly(img_lung_ch, [contours_lung_ch], (255))
    cv2.drawContours(img_lung_ch, [contours_lung_ch], -1, (255), -1)

    saveimg(pathoux, img_lung_ch, ix, i+"_4.3", PSAVE)


    ##  5) save results
    ##  x) draw results
    img = np.stack((imgraw,)*3, axis=-1)
    ### x) draw lung countours 
    cv2.drawContours(img, contours_lung, -1, (0,255,0), 1)
    ### x) draw lung ch contours
    cv2.drawContours(img, [contours_lung_ch], -1, (255, 0, 0), 2)
    saveimg(pathoux, img, ix, i+"_5", PSAVE)




    return img_largest_fill_holles, img_lung, img_lung_ch, contours_thorax, contours_lung, contours_lung_ch



def createsplit(path, p=0.1, cut=100):
    makedir(os.path.join(path, "split"))
    
    makedir(os.path.join(path, "split", "train"))
    makedir(os.path.join(path, "split", "train", "groundtruths"))
    makedir(os.path.join(path, "split", "train", "images"))


    makedir(os.path.join(path, "split", "test"))
    makedir(os.path.join(path, "split", "test", "groundtruths"))
    makedir(os.path.join(path, "split", "test", "images"))


    pathroot = os.path.join(path, "images")
    pathgroundtruth = os.path.join(path, "groundtruth")

    pathout = os.path.join(path, "split")


    # pathroot_fils = [im for im in os.listdir(os.path.join(pathroot, "roi_lung_ch_radiologists"))]
    pathroot_fils = [im for im in os.listdir(os.path.join(pathgroundtruth, "mask"))]
    
    
    imgs = {}
    for fi in pathroot_fils:
        idname = fi.split("_")
        imgs[idname[0]] = 0

    for fi in pathroot_fils:
        idname = fi.split("_")
        im_msk = cv2.imread(os.path.join(pathgroundtruth, "mask", fi),-1)

        # cc = im_msk[np.where(im_msk>0)]
        # areamm(len(cc[0]), data.PixelSpacing)

        # print("fi", im_msk)
        imgs[idname[0]] += 1

    # imgsorder = sorted(imgs)
    imgsor = sorted(imgs.items(), key=lambda x: x[1])
    imgsorder = []
    for dd in imgsor:
        imgsorder.append(dd[0])
    # print("imgsorder", imgsorder)
    if len(imgsorder)>cut:
        imgsorder = imgsorder[-cut:]
    for ii in imgsorder:
        print("QQ: ", ii, imgs[ii])

    print("len(imgsorder)", len(imgsorder))
    s = int(len(imgsorder)*p)
    testi = imgsorder[0:s]
    traini = imgsorder[s:]
    print(testi)
    print(traini)
    exit(0)
    
    # images
    pathimages = [  "roi_lung_ch_CLAHE",
                    "roi_lung_ch_manualwindowing",
                    "roi_lung_ch_radiologists",
                    "roi_lung_ch_windowing",
                    "roi_thorax_CLAHE",
                    "roi_thorax_manualwindowing",
                    "roi_thorax_radiologists",
                    "roi_thorax_windowing",
                    ]

    for pi in pathimages:
        pthi = os.path.join(pathroot, pi)
        for fi in os.listdir(pthi):
            idname = fi.split("_")
            idname=idname[0]
            # print("idname", idname)
            if idname in testi:
                ptho = os.path.join(pathout, "test", "images", pi)
                makedir(ptho)

                shutil.copyfile(os.path.join(pthi, fi), os.path.join(ptho, fi))
            elif idname in traini:
                ptho = os.path.join(pathout, "train", "images", pi)
                makedir(ptho)

                shutil.copyfile(os.path.join(pthi, fi), os.path.join(ptho, fi))

    # groundtruths
    pathimages = ["roi_lung_ch", "roi_thorax"]
    for pi in pathimages:
        pthi = os.path.join(pathgroundtruth, pi)
        for fi in os.listdir(pthi):
            idname = fi.split("_")
            idname=idname[0]
            if idname in testi:
                ptho = os.path.join(pathout, "test", "groundtruths", pi)
                makedir(ptho)

                shutil.copyfile(os.path.join(pthi, fi), os.path.join(ptho, fi))
            elif idname in traini:
                ptho = os.path.join(pathout, "train", "groundtruths", pi)
                makedir(ptho)

                shutil.copyfile(os.path.join(pthi, fi), os.path.join(ptho, fi))


def splitDData(ds, version, p=0.1):

    # data/lugcancer/clean/v2/
    # "primarias", "20240717184542"
    path = os.path.join(pathou, ds, version)

    ptht = os.path.join(pathou, ds, version, "images", "whole_dcm")
    pthtmks = os.path.join(pathou, ds, version, "groundtruth", "mask")
    
    print("ptht", ptht)
    datax = {}
    for ix in os.listdir(ptht):
        datax[ix] = {"case":ix, "type":"", "slides":0,
                "PixelSpacing_x":0.0, "PixelSpacing_y":0.0,
                "area":0.0, "area_nodule": 0.0,
                "images": [],
                }

    for ix in os.listdir(ptht):
        slidir = os.listdir(os.path.join(ptht, ix))
        for pdir in slidir:
            fraw = os.path.join(ptht, ix, pdir)
            # print("fraw", fraw)
            data = dcm.dcmread(fraw)
            # ### begin compute pixelspacing
            ssx, ssy = data.PixelSpacing
            # print("x x PixelSpacing.PixelSpacing", ssx, ssy)


            file_name = os.path.splitext(os.path.basename(fraw))[0]
            im_mks = cv2.imread(os.path.join(pthtmks, file_name+".png"),-1)
            # print("im_mks", im_mks)

            cc = np.where(im_mks>0)
            areai = areamm(len(cc[0]), data.PixelSpacing)
            arealimiinf = 3.14159265*((7.5)**2)
            arealinode = 3.14159265*((15.0)**2)
            # print("areai", areai, arealimiinf)
            # 7.5*2 = 15 = 30/2 of node diameter
            # arealimiinf = 3.14159265*((7.5)**2)
            tipo = "nodule"
            if areai>arealinode:
                tipo = "mass"

            if areai > datax[ix]["area"]:
                datax[ix]["area"] = areai 
                datax[ix]["type"] = tipo
            datax[ix]["area_nodule"] = arealinode 
            datax[ix]["PixelSpacing_x"] = ssx 
            datax[ix]["PixelSpacing_y"] = ssy 
            datax[ix]["images"].append(file_name+".png")

        datax[ix]["slides"] = len(slidir)
    

    dattt = []
    for rr in datax:
        if datax[rr]["slides"]>=4:# seleccionar casos com 4 ou mais slides
            dattt.append(datax[rr])
        # print("datax", rr, datax[rr])

    newlist = sorted(dattt, key=operator.itemgetter('slides'))

    # if ds=="metastase":
    #     newlist = newlist[-50:]
    # if ds=="primarias":
    #     newlist = newlist[-100:]

    ccnodule = []
    ccmass = []
    for rr in newlist:
        print("datax", rr)
        if rr["type"]=="nodule":
            ccnodule.append(rr["case"])
        elif rr["type"]=="mass":
            ccmass.append(rr["case"])
    print("ccmass, ccnodule", len(ccmass), len(ccnodule))

    # choose cases
    # ############ rand
    random.seed(10)
    random.shuffle(ccmass)
    random.shuffle(ccnodule)

    # ############ sort
    # ccmass = sorted(ccmass, key=operator.itemgetter('slides'))
    # ccnodule = sorted(ccnodule, key=operator.itemgetter('slides'))

    print("len(ccmass), len(ccmass)", len(ccmass), len(ccmass))

    s1 = round(len(ccmass)*p)
    s2 = round(len(ccnodule)*p)

    testi = ccmass[:s1]
    traini = ccmass[s1:]
    testi += ccnodule[:s2]
    traini += ccnodule[s2:]



    ################
    ####### split ##
    ################

    makedir(os.path.join(path, "split"))
    
    makedir(os.path.join(path, "split", "train"))
    makedir(os.path.join(path, "split", "train", "groundtruths"))
    makedir(os.path.join(path, "split", "train", "images"))


    makedir(os.path.join(path, "split", "test"))
    makedir(os.path.join(path, "split", "test", "groundtruths"))
    makedir(os.path.join(path, "split", "test", "images"))


    pathroot = os.path.join(path, "images")
    pathgroundtruth = os.path.join(path, "groundtruth")

    pathout = os.path.join(path, "split")


    # images
    pathimages = [  
                    # {"images":"roi_thorax_CLAHE", "masks":"roi_thorax"},
                    # {"images":"roi_thorax_manualwindowing", "masks":"roi_thorax"},
                    # {"images":"roi_thorax_radiologists", "masks":"roi_thorax"},
                    # {"images":"roi_thorax_windowing", "masks":"roi_thorax"},

                    # {"images":"roi_lung_ch_CLAHE", "masks":"roi_lung_ch"},
                    # {"images":"roi_lung_ch_manualwindowing", "masks":"roi_lung_ch"},
                    # {"images":"roi_lung_ch_radiologists", "masks":"roi_lung_ch"},
                    # {"images":"roi_lung_ch_windowing", "masks":"roi_lung_ch"},
        
                    # {"images":"roi_lung_CLAHE", "masks":"roi_lung"},
                    {"images":"roi_lung_manualwindowing", "masks":"roi_lung"},
                    # {"images":"roi_lung_radiologists", "masks":"roi_lung"},
                    # {"images":"roi_lung_windowing", "masks":"roi_lung"},
                ]
    if True:

        for ddat in pathimages:
            pi_img = ddat["images"]
            pi_maks = ddat["masks"]
            pth_img = os.path.join(pathroot, pi_img)
            pth_mask = os.path.join(pathgroundtruth, pi_maks)
            for fi in os.listdir(pth_img):
                idname = fi.split("_")
                idname=idname[0]
                # print("idname", idname)
                if idname in testi:
                    ptho = os.path.join(pathout, "test", "images", pi_img)
                    makedir(ptho)
                    shutil.copyfile(os.path.join(pth_img, fi), os.path.join(ptho, fi))

                    ptho = os.path.join(pathout, "test", "groundtruths", pi_maks)
                    makedir(ptho)
                    shutil.copyfile(os.path.join(pth_mask, fi), os.path.join(ptho, fi))

                elif idname in traini:
                    ptho = os.path.join(pathout, "train", "images", pi_img)
                    makedir(ptho)
                    shutil.copyfile(os.path.join(pth_img, fi), os.path.join(ptho, fi))

                    ptho = os.path.join(pathout, "train", "groundtruths", pi_maks)
                    makedir(ptho)
                    shutil.copyfile(os.path.join(pth_mask, fi), os.path.join(ptho, fi))


    #####################################
    #### data augmentation ##############
    #####################################
    augmentedtrain = "train_augmented"
    makedir(os.path.join(pathout, augmentedtrain))



    traini_rra = []
    testi_rra = []
    for ttt in traini:
        for rr in newlist:
            if ttt == rr["case"]:
                traini_rra.append(rr)

    for ttt in testi:
        for rr in newlist:
            if ttt == rr["case"]:
                testi_rra.append(rr)


    print("test, train", len(testi), len(traini))
    print("test, train", testi, traini)
    print("testi_rra, traini_rra", len(testi_rra), len(traini_rra) )
    print("testi_rra, traini_rra", testi_rra, traini_rra )


    # mm = int(len(traini_rra)/2)
    
    traini_rra_sorted = sorted(traini_rra, key=operator.itemgetter('slides'))
    # traini_rra_sorted = traini_rra_sorted_x[:mm]

    maxslices = (traini_rra_sorted[-1]["slides"])
    
    traini_rra_sorted_names = [] 
    for ddd in traini_rra_sorted:
        traini_rra_sorted_names.append(ddd["case"])
        print("ABC", maxslices, ddd["case"], ddd["slides"])

    
    for ddat in pathimages:
        pi_img = ddat["images"]
        pi_maks = ddat["masks"]
        pth_img = os.path.join(pathroot, pi_img)
        pth_mask = os.path.join(pathgroundtruth, pi_maks)

        makedir(os.path.join(pathout, augmentedtrain, pi_maks))
        makedir(os.path.join(pathout, augmentedtrain, pi_maks, pi_img))
        makedir(os.path.join(pathout, augmentedtrain, pi_maks, pi_img, "images"))
        makedir(os.path.join(pathout, augmentedtrain, pi_maks, pi_img, "groundtruths"))

            

        for dirname in traini_rra_sorted_names:
            nslices = datax[dirname]["slides"]
            
            # ninter = int(round(maxslices/2.0)-nslices)
            ninter = int(round(maxslices)-nslices)
            
            if ninter>0:
                iii=0
                # print("datax[dirname][images]", datax[dirname]["images"])
                pool = cycle(datax[dirname]["images"])
                for fi in pool:
                    iii+=1
                    print("fi", fi, iii)
                    
                    file_name = os.path.splitext(os.path.basename(fi))[0]
                    file_name = file_name.split("_")[1]

                    im_gray = cv2.imread(os.path.join(pth_img, fi),-1)
                    im_mask = cv2.imread(os.path.join(pth_mask, fi),-1)

                    # draw_grid(im_gray, 20)
                    # draw_grid(im_mask, 20)

                    im_gray_t, im_mask_t = elastic_transform(im_gray, im_mask,
                            # im.shape[1] * 0.19,
                            # im.shape[1] * 0.19,
                            # im.shape[1] * 0.001,
                            im_gray.shape[1]*1.0,
                            im_gray.shape[1]*0.038,
                            im_gray.shape[1]*0.0
                            )
                            # random_state=np.random.RandomState(42)

                    # print("fimg, fmask", im_gray, im_mask)

                    ptho_i = os.path.join(pathout, augmentedtrain, pi_maks, pi_img, "images", dirname+"_"+str(iii)+"_"+file_name+".png")
                    ptho_m = os.path.join(pathout, augmentedtrain, pi_maks, pi_img, "groundtruths", dirname+"_"+str(iii)+"_"+file_name+".png")
                    # print("ptho_i, ptho_m", ptho_i, ptho_m)
                    cv2.imwrite(ptho_i, im_gray_t)
                    cv2.imwrite(ptho_m, im_mask_t)
                    if iii==ninter:
                        break

                # if idname in testi_rra_sorted_names:


#########3https://towardsdatascience.com/medical-image-pre-processing-with-python-d07694852606
#MEJORAS EN LA IMAGEN#
#get_pixels_hu()
#read_dir_dicom()





# aug_combine()

limiinf = 3.14159265*((7.5)**2)
# # # # # # limiinf = 3.14159265*((5.5)**2)
# readrr("primarias", limiinf,
# ["P8","P16","P91","P101","P126","P127","P1","P29","P116",
# "P1","P23","P30","P121","P124","P32","P33","P114","P51","P56","P57","P93","P97","P103",
# "P68","P85","P93","P89","P92","P96","P92","P4","P5","P95","P55",

# "P109","P128","P17","P24","P12","P11","P19","P31","P36",
# "P184","P185","P47","P48","P58","P59","P61","P9"])
# readrr("metastase", limiinf, ["P24","P30"])

splitDData("primarias", "20240919095641", p=0.1)
# splitDData("metastase", "20240917015723", p=0.1)

# 20240917015723 v5 (metastase)
# 20240919095641 v5 (primaries)




##### * * train and test split * *
# pp = "../../../../data/lugcancer/clean/v2/primarias/20231201011853"
# createsplit(pp,0.1)
# pp = "../../../../data/lugcancer/clean/v2/metastase/20231201013504"
# createsplit(pp,0.1)
#######
# pp = "../../../../data/lugcancer/clean/v2/primarias/20240717184542"
# createsplit(pp,0.1, 100)
# pp = "../../../../data/lugcancer/clean/v2/metastase/20240717191502"
# createsplit(pp, 0.1, 50)




# readrr("primarias", -40)
# readrr("primarias", 40)
# creasplit(pathoux, 132)


# # DS2
# # #22
# # for i in range(1,112):
# #     readrr("ds2",str(i))
# pathoux = os.path.join(pathou, "ds2", now())
# makedir(pathoux)
# for i in range(1,112):
#     #if i!=63 or i!=87:
#     readrr("ds2",str(i), pathoux, 400)
# creasplit(pathoux, 112)






















""" 
#read_dir_dicom()
img = get_pixels_hu()
print("img", img)

img = ( (img - img.min() ) / (img.max()-img.min() )) * 255.0
img = np.uint8(img)

cv2.imwrite('out_pixels.png', img)

"""