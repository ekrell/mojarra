#!/usr/bin/python3 
import numpy as np
import time
from optparse import OptionParser
import cv2

####################
# Color correction #
####################

def white_cb(img):
    sb = cv2.xphoto.createSimpleWB()
    ret = sb.balanceWhite(img)
    return ret

def simplest_cb(img, percent = 1):
    # Source: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    # "Simplest Color Balance", same as Photoshop's "auto levels" command
    # Jason Su
    out_channels = []
    channels = cv2.split(img)
    totalstop = channels[0].shape[0] * channels[0].shape[1] * percent / 200.0
    for channel in channels:
        bc = cv2.calcHist([channel], [0], None, [256], (0,256), accumulate=False)
        lv = np.searchsorted(np.cumsum(bc), totalstop)
        hv = 255-np.searchsorted(np.cumsum(bc[::-1]), totalstop)
        lut = np.array([0 if i < lv else (255 if i > hv else round(float(i-lv)/float(hv-lv)*255)) for i in np.arange(0, 256)], dtype="uint8")
        out_channels.append(cv2.LUT(channel, lut))
    return cv2.merge(out_channels)

def grayworld_cb(img, threshold = 0.95):
    # Default OpenCV color balancing based on gray world assumption
    gw = cv2.xphoto.createGrayworldWB()
    gw.setSaturationThreshold(threshold)
    ret = gw.balanceWhite(img)
    return ret

def grayworldLAB_cb(img):
    # Source: https://stackoverflow.com/a/46391574
    # Apply the greyworld assumption in LAB space (A, B channels )
    # https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html
    res = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(res[:, :, 1])
    avg_b = np.average(res[:, :, 2])
    res[:, :, 1] = res[:, :, 1] - ((avg_a - 128) * (res[:, :, 0] / 255.0) * 1.1)
    res[:, :, 2] = res[:, :, 2] - ((avg_b - 128) * (res[:, :, 0] / 255.0) * 1.1)
    res = cv2.cvtColor(res, cv2.COLOR_LAB2BGR)
    return res

def satintBalance(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    res[:, :, 2] = cv2.equalizeHist(res[:, :, 2])
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    return res

def clahe(img, clipLimit = 2.0, tileGridSize = (4, 4)):
    ret = img.copy()
    clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)
    for i in range(3):
        ret[:, :, i] = clahe.apply((img[:, :, i]))
    return ret

def gammaCorrection(img, p = 0.9):
    ret = img.copy()
    ret = ret / 255.0
    for i in range(3):
        ret[:, :, i] = np.power(ret[:, :, i] / float(np.max(ret[:, :, i])), p)
    ret = np.clip(ret * 255, 0, 255)
    ret = np.uint8(ret)
    return ret

def enhance(img, color_balance = None, lighting = None):
    ret = img.copy()

    if color_balance is not None:
        if color_balance == "white":
            ret = white_cb(ret)
        elif color_balance == "simple":
            ret = simple_cb(ret, percent = 1)
        elif color_balance == "gray":
            ret = grayworld_cb(ret)
        elif color_balance == "graylab":
            ret = grayworldLAB_cb

    return ret

def main():

    # Options
    parser = OptionParser()
    parser.add_option("-f", "--files",
            help = "Path to file where each line is a path to an image file to process.")
    parser.add_option("-o", "--outdir",
            help = "Path to directory to store processed output.")
    parser.add_option("-s", "--show",
            help = "Show each enhanced image.",
            default = False, action = "store_true")
    (options, args) = parser.parse_args()
    if options.files is None:
        print("Option '--files' ('-f') required!")
        exit(-1)
    if options.outdir is None:
        print("Option '--outdir' ('-o') required!")
        exit(-1)

    # Read list of images
    with open(options.files) as f:
        imgPaths = f.read().splitlines()

    # Initialize feature detectors
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)

    # Process each image
    for imgPath in imgPaths:
        outPath = options.outdir + "/" + imgPath.split("/")[-1]
        orig = cv2.imread(imgPath)
        
        # Enhance
        enhanced = enhance(orig, color_balance = "white")

        # Extract features
        keys_sift_orig, descriptors = sift.detectAndCompute(orig, None)
        keys_surf_orig, descriptors = surf.detectAndCompute(orig, None)
        keys_orb_orig, descriptors = orb.detectAndCompute(orig, None)
        keys_sift_enhanced, descriptors = sift.detectAndCompute(enhanced, None)
        keys_surf_enhanced, descriptors = surf.detectAndCompute(enhanced, None)
        keys_orb_enhanced, descriptors = orb.detectAndCompute(enhanced, None)


        # Report
        print("Image: {}".format(imgPath))
        print("   Original features: {} SIFT, {} SURF, {} ORB".format(
             len(keys_sift_orig), len(keys_surf_orig), len(keys_orb_orig)))
        print("   Enhanced features: {} SIFT, {} SURF, {} ORB".format(
             len(keys_sift_enhanced), len(keys_surf_enhanced), len(keys_orb_enhanced)))


        # Show before and after processing
        if options.show:
            origKeys = cv2.drawKeypoints(orig, keys_surf_orig, None)
            enhancedKeys = cv2.drawKeypoints(enhanced, keys_surf_enhanced, None)
            cv2.imshow('Comparison', np.hstack((origKeys, enhancedKeys)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Write processed image
        cv2.imwrite(outPath, enhanced)

if __name__ == '__main__':
    main()
