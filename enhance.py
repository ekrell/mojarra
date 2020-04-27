#!/usr/bin/python3 
import numpy as np
import time
from optparse import OptionParser
import matplotlib.pyplot as plt
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
            help = "Show each enha image.",
            default = False, action = "store_true")
    parser.add_option("-m", "--match",
            help = "Match features across image pairs.",
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
    # Initialize BFMatcher object
    bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    origPrev = None
    enhaPrev = None
    # Process each image
    for imgPath in imgPaths:
        outPath = options.outdir + "/" + imgPath.split("/")[-1]
        orig = cv2.imread(imgPath)
        
        # Enhance
        enha = enhance(orig, color_balance = "white")

        # Extract features
        keys_sift_orig, desc_sift_orig = sift.detectAndCompute(orig, None)
        keys_surf_orig, desc_surf_orig = surf.detectAndCompute(orig, None)
        keys_orb_orig, desc_orb_orig = orb.detectAndCompute(orig, None)
        keys_sift_enha, desc_sift_enha = sift.detectAndCompute(enha, None)
        keys_surf_enha, desc_surf_enha = surf.detectAndCompute(enha, None)
        keys_orb_enha, desc_orb_enha = orb.detectAndCompute(enha, None)

        # Report
        print("Image: {}".format(imgPath))
        print("   Original features: {} SIFT, {} SURF, {} ORB".format(
             len(keys_sift_orig), len(keys_surf_orig), len(keys_orb_orig)))
        print("   Enha features: {} SIFT, {} SURF, {} ORB".format(
             len(keys_sift_enha), len(keys_surf_enha), len(keys_orb_enha)))


        # Write processed image
        cv2.imwrite(outPath, enha)

        # Test feature matching
        if options.match:
            if origPrev is not None:
                # SIFT matches
                matchesOrig_sift = bf_sift.match(desc_sift_origPrev, desc_sift_orig)
                matchesEnha_sift = bf_sift.match(desc_sift_enhaPrev, desc_sift_enha)
                # Sort them in the order of their distance.
                matchesOrig_sift = sorted(matchesOrig_sift, key = lambda x:x.distance)
                matchesEnha_sift = sorted(matchesEnha_sift, key = lambda x:x.distance)
                # Apply match test
                matchesOrigGood_sift = []
                for i, m in enumerate(matchesOrig_sift):
                    if i < len(matchesOrig_sift) - 1 and m.distance < 0.97 * matchesOrig_sift[i + 1].distance:
                        matchesOrigGood_sift.append(m)
                matchesOrig_sift = matchesOrigGood_sift
                matchesEnhaGood_sift = []
                for i, m in enumerate(matchesEnha_sift):
                    if i < len(matchesEnha_sift) - 1 and m.distance < 0.97 * matchesEnha_sift[i + 1].distance:
                        matchesEnhaGood_sift.append(m)
                matchesEnha_sift = matchesEnhaGood_sift

                # SURF matches
                matchesOrig_surf = bf_sift.match(desc_surf_origPrev, desc_surf_orig)
                matchesEnha_surf = bf_sift.match(desc_surf_enhaPrev, desc_surf_enha)
                # Sort them in the order of their distance.
                matchesOrig_surf = sorted(matchesOrig_surf, key = lambda x:x.distance)
                matchesEnha_surf = sorted(matchesEnha_surf, key = lambda x:x.distance)
                # Apply match test
                matchesOrigGood_surf = []
                for i, m in enumerate(matchesOrig_surf):
                    if i < len(matchesOrig_surf) - 1 and m.distance < 0.97 * matchesOrig_surf[i + 1].distance:
                        matchesOrigGood_surf.append(m)
                matchesOrig_surf = matchesOrigGood_surf
                matchesEnhaGood_surf = []
                for i, m in enumerate(matchesEnha_surf):
                    if i < len(matchesEnha_surf) - 1 and m.distance < 0.97 * matchesEnha_surf[i + 1].distance:
                        matchesEnhaGood_surf.append(m)
                matchesEnha_surf = matchesEnhaGood_surf

                # ORB matches
                matchesOrig = bf_orb.match(desc_orb_origPrev, desc_orb_orig)
                matchesEnha = bf_orb.match(desc_orb_enhaPrev, desc_orb_enha)
                # Sort them in the order of their distance.
                matchesOrig = sorted(matchesOrig, key = lambda x:x.distance)
                matchesEnha = sorted(matchesEnha, key = lambda x:x.distance)
                # Apply match test
                matchesOrigGood = []
                for i, m in enumerate(matchesOrig):
                    if i < len(matchesOrig) - 1 and m.distance < 0.97 * matchesOrig[i + 1].distance:
                        matchesOrigGood.append(m)
                matchesOrig = matchesOrigGood
                matchesEnhaGood = []
                for i, m in enumerate(matchesEnha):
                    if i < len(matchesEnha) - 1 and m.distance < 0.97 * matchesEnha[i + 1].distance:
                        matchesEnhaGood.append(m)
                matchesEnha = matchesEnhaGood

                print("   Original matches: {} SIFT, {} SURF, {} ORB".format(
                    len(matchesOrig_sift), len(matchesOrig_surf), len(matchesOrig)))
                print("   Enha matches: {} SIFT, {} SURF, {} ORB".format(
                    len(matchesEnha_sift), len(matchesEnha_surf), len(matchesEnha)))


        # Show before and after processing
        if options.show and origPrev is not None:
            fig, ax = plt.subplots(figsize=(20, 10))

            origKeys_sift = cv2.drawKeypoints(orig, keys_sift_orig, None)
            enhaKeys_sift = cv2.drawKeypoints(enha, keys_sift_enha, None)
            origKeys_surf = cv2.drawKeypoints(orig, keys_surf_orig, None)
            enhaKeys_surf = cv2.drawKeypoints(enha, keys_surf_enha, None)
            origKeys_orb = cv2.drawKeypoints(orig, keys_orb_orig, None)
            enhaKeys_orb = cv2.drawKeypoints(enha, keys_orb_enha, None)
            
            if options.match:
                matchesOrigImg_sift = cv2.drawMatches(origPrev, keys_sift_origPrev, 
                        orig, keys_sift_orig, matchesOrig_sift, None, flags = 2)
                matchesEnhaImg_sift = cv2.drawMatches(enhaPrev, keys_sift_enhaPrev, 
                        enha, keys_sift_enha, matchesEnha_sift, None, flags = 2)
                matchesOrigImg_surf = cv2.drawMatches(origPrev, keys_surf_origPrev, 
                        orig, keys_surf_orig, matchesOrig_surf, None, flags = 2)
                matchesEnhaImg_surf = cv2.drawMatches(enhaPrev, keys_surf_enhaPrev, 
                        enha, keys_sift_enha, matchesEnha_sift, None, flags = 2)
                matchesOrigImg_orb = cv2.drawMatches(origPrev, keys_orb_origPrev, 
                        orig, keys_orb_orig, matchesOrig, None, flags = 2)
                matchesEnhaImg_orb = cv2.drawMatches(enhaPrev, keys_orb_enhaPrev, 
                        enha, keys_orb_enha, matchesEnha, None, flags = 2)

                ax.imshow(np.vstack(( \
                    np.hstack((origKeys_sift, enhaKeys_sift, matchesOrigImg_sift, matchesEnhaImg_sift)),
                    np.hstack((origKeys_surf, enhaKeys_surf, matchesOrigImg_surf, matchesEnhaImg_surf)),
                    np.hstack((origKeys_orb, enhaKeys_orb, matchesOrigImg_orb, matchesEnhaImg_orb)))), 
                    interpolation = 'nearest')

                
            ax.set_aspect('auto')
            plt.show()
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


        # Set current to previous
        origPrev = orig.copy()
        enhaPrev = enha.copy()
        keys_sift_origPrev = keys_sift_orig 
        keys_surf_origPrev = keys_surf_orig
        keys_orb_origPrev = keys_orb_orig
        keys_sift_enhaPrev = keys_sift_enha
        keys_surf_enhaPrev = keys_surf_enha
        keys_orb_enhaPrev = keys_orb_enha
        desc_sift_origPrev = desc_sift_orig 
        desc_surf_origPrev = desc_surf_orig
        desc_orb_origPrev = desc_orb_orig
        desc_sift_enhaPrev = desc_sift_enha
        desc_surf_enhaPrev = desc_surf_enha
        desc_orb_enhaPrev = desc_orb_enha

if __name__ == '__main__':
    main()
