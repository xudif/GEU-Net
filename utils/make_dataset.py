import os
import csv


def create_csv(base_path, img1, img2):
    img1_path = base_path + img1 + '/'  # Clear foreground image path
    img2_path = base_path + img2 + '/'  # Clear background image path
    label_path = './mask/'  # Corresponding mask path
    img1_name = os.listdir(img1_path)
    img2_name = os.listdir(img2_path)
    label_name = os.listdir(label_path)
    img1_name.sort(key=lambda x: int(x[:-4]))
    img2_name.sort(key=lambda x: int(x[:-4]))
    label_name.sort(key=lambda x: int(x[:-4]))

    with open('all_train_img' + '.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for n in range(0, len(img1_name)):
            sub_img1_name = img1_name[n]
            sub_img2_name = img2_name[n]
            sub_label_name = label_name[n]
            if sub_img1_name[-4:] == '.png':
                print(sub_img1_name)
                writer.writerow([base_path + str(img1) + '/' + str(sub_img1_name),
                                 base_path + str(img2) + '/' + str(sub_img2_name),
                                 './mask/' + str(sub_label_name)])
            else:
                pass


if __name__ == "__main__":
    create_csv('./result/blur1/', str(1), str(2))
