import sys
import numpy as np
import pydensecrf.densecrf as dcrf
import cv2,os
import re

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

from PIL import Image

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary


def crf(x, y, z):
    fn_im = x
    fn_anno = y
    fn_output = z

    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR

    anno_rgb = imread(fn_anno).astype(np.uint32)
    # anno_rgb = anno_rgb.astype(np.uint32)

    # anno_rgb = anno_rgb.astype(np.uint32)
    anno_rgb[anno_rgb < 1] = 1
    anno_rgb[anno_rgb > 1] = 255

    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print(
            "Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print(
            "If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    # else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        sdims1 = 5
        feats = create_pairwise_gaussian(sdims=(sdims1, sdims1), shape=img.shape[:2])  # 3
        d.addPairwiseEnergy(feats, compat=5,  # 3
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        sdims = 110
        schan = 13
        feats = create_pairwise_bilateral(sdims=(sdims, sdims), schan=(schan, schan, schan),  # 50,20
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=6,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    # imwrite(fn_output, MAP.reshape(img.shape))
    imwrite(fn_output, MAP.reshape(imread(fn_anno).shape))

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    d.stepInference(Q, tmp1, tmp2)

    print(np.shape(Q), np.shape(MAP), np.shape(tmp2))
