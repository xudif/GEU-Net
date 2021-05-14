import numpy as np
from skimage import io, morphology
from utils.CRF import crf

def fusion(initial_decisionmap_path, final_decisionmap_path, image1_path, image2_path, initial_image_save_path, fusion_image_save_path):
    global initial_decisionmap, final_decisionmap
    initial_decisionmap = io.imread(initial_decisionmap_path)
    initial_decisionmap = initial_decisionmap / 255.0
    initial_decisionmap = np.array(initial_decisionmap)
    initial_decisionmap[initial_decisionmap <= 0.5] = 0
    initial_decisionmap[initial_decisionmap > 0.5] = 1

    decisionmap = io.imread(final_decisionmap_path)
    decisionmap = decisionmap / 255.0
    decisionmap = np.array(decisionmap)
    decisionmap[decisionmap < 0.5] = 0
    decisionmap[decisionmap > 0.5] = 1
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)

    if len(np.array(image1).shape) == 2:
        if len(np.array(decisionmap).shape) == 3:
            final_decisionmap = decisionmap[:, :, 0]
            fusion_image = image1 * final_decisionmap + image2 * (1 - final_decisionmap)
            # save fused image
            io.imsave(fusion_image_save_path, fusion_image.astype(np.uint8))
        else:
            final_decisionmap = decisionmap
            fusion_image = image1 * final_decisionmap + image2 * (1 - final_decisionmap)
            # save fused image
            io.imsave(fusion_image_save_path, fusion_image.astype(np.uint8))

        initial_fusion_image = image1 * initial_decisionmap + image2 * (1 - initial_decisionmap)
        io.imsave(initial_image_save_path, initial_fusion_image.astype(np.uint8))

    else:
        final_decisionmap = np.ones([image1.shape[0], image1.shape[1], image1.shape[2]])
        print("init: ", initial_decisionmap.shape)
        initial_decisionmap = np.expand_dims(initial_decisionmap, 2)
        print("1: ", initial_decisionmap.shape)
        initial_decisionmap = np.repeat(initial_decisionmap, 3, axis=2)
        print("2: ", initial_decisionmap.shape)
        if len(np.array(decisionmap).shape) == 2:
            final_decisionmap[:, :, 0] = decisionmap
            final_decisionmap[:, :, 1] = decisionmap
            final_decisionmap[:, :, 2] = decisionmap
        else:
            final_decisionmap = decisionmap

        fusion_image = image1 * final_decisionmap + image2 * (1 - final_decisionmap)
        # save fused image
        io.imsave(fusion_image_save_path, fusion_image.astype(np.uint8))

        # save initial fused image
        initial_fusion_image = image1 * initial_decisionmap + image2 * (1 - initial_decisionmap)
        io.imsave(initial_image_save_path, initial_fusion_image.astype(np.uint8))


def fusion_main(initial_decisionmap_path, image1_path, image2_path, final_decisionmap_save_path, final_crf_path, initial_image_save_path, fusion_image_save_path):

    # CRF
    crf(image1_path, initial_decisionmap_path, final_crf_path)

    # fusion
    fusion(initial_decisionmap_path, final_crf_path, image1_path, image2_path, initial_image_save_path, fusion_image_save_path)


def M_fusion(version, type):
    n = 1
    for i in range(1, 21):
        print(i)
        if n < 10:
            initial_decisionmap_path = "./image/test_pred/" + version + "/" + type + "/initial_decisionmap/0" + str(i) + ".png"
            image1_path = "./data/test/image1/lytro-0" + str(n) + "-A.jpg"
            image2_path = "./data/test/image2/lytro-0" + str(n) + "-B.jpg"
            final_decisionmap_save_path = "./result/" + version + "/" + type + "/decisionMap_lytro_" + str(i) + "_crf.png"
            final_crf_path = "./result/" + version + "/" + type + "/decisionmap_crf/lytro_" + str(i) + "_crf.png"
            initial_image_save_path = "./result/" + version + "/" + type + "/initial_fusion/lytro_" + str(i) + ".png"
            fusion_image_save_path = "./result/" + version + "/" + type + "/fusion/lytro_" + str(i) + "_crf.png"
            fusion_main(initial_decisionmap_path, image1_path, image2_path, final_decisionmap_save_path, final_crf_path,
                        initial_image_save_path, fusion_image_save_path)
            n += 1
        else:
            initial_decisionmap_path = "./image/test_pred/" + version + "/" + type + "/initial_decisionmap/" + str(i) + ".png"
            image1_path = "./data/test/image1/lytro-" + str(n) + "-A.jpg"
            image2_path = "./data/test/image2/lytro-" + str(n) + "-B.jpg"
            final_decisionmap_save_path = "./result/" + version + "/" + type + "/decisionMap_lytro_" + str(i) + "_crf.png"
            final_crf_path = "./result/" + version + "/" + type + "/decisionmap_crf/lytro_" + str(i) + "_crf.png"
            initial_image_save_path = "./result/" + version + "/" + type + "/initial_fusion/lytro_" + str(i) + ".png"
            fusion_image_save_path = "./result/" + version + "/" + type + "/fusion/lytro_" + str(i) + "_crf.png"
            fusion_main(initial_decisionmap_path, image1_path, image2_path, final_decisionmap_save_path, final_crf_path,
                        initial_image_save_path, fusion_image_save_path)
            n += 1