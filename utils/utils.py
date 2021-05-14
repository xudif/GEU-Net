import argparse
import os
import torchvision.models as models
from model.unet_model import *

def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(prog='GEU-Net params',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="in and out folder")
    parser.add_argument('-s', '--state', type=str, default="train",
                        help=r' 2 state train, inference')
    parser.add_argument('-t', '--train_csv', type=str, default='./all_train_img.csv',
                        help=r'training dataset csv file')
    parser.add_argument('-v', '--val', type=str, default="./data/valid",
                        help=r'path to validate folder')
    parser.add_argument('-te', '--test', type=str, default="./data/test",
                        help=r'path to test folder')
    parser.add_argument('-st', '--step2', type=bool, default=False,
                        help=r'whether the second stage')
    parser.add_argument('-m', '--model', type=str, default='GEU_Net',
                        help=r'choose training model')
    parser.add_argument('-w', '--weights', type=str, default='_model/cp_20_14000_0.0185030996799469.pth',
                        help=r'path to GEU-Net weights')
    parser.add_argument('-btr', '--batchsize_train', type=int, default=16,
                        help=r'number of images, simultaneously sent to the GPU')
    parser.add_argument('-bv', '--batchsize_valid', type=int, default=16,
                        help=r'number of images, simultaneously sent to the GPU')
    parser.add_argument('-bte', '--batchsize_test', type=int, default=1,
                        help=r'number of images, simultaneously sent to the GPU')
    parser.add_argument('-g', '--gpus', type=str, default="0",
                        help=r'number of GPUs for binarization')
    parser.add_argument('-r', '--lr', type=float, default=0.0005)
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-w1', '--w1_bce', type=int, default=1,
                        help=r'weights for BCELoss')
    parser.add_argument('-w2', '--w2_per', type=int, default=1,
                        help=r'weights for PerceptualLoss')
    parser.add_argument("-mi", "--model_inchannel", type=int, default=2)
    parser.add_argument("-mo", "--model_outchannel", type=int, default=2)
    return parser.parse_args()

class PerceptualLoss(nn.Module):
    def __init__(self, is_cuda):
        super(PerceptualLoss, self).__init__()
        print('loading resnet101...')
        self.loss_network = models.resnet101(pretrained=True, num_classes=2, in_channel=2)
        # Turning off gradient calculations for efficiency
        for param in self.loss_network.parameters():
            param.requires_grad = False
        if is_cuda:
            self.loss_network.cuda()
        print("done ...")

    def mse_loss(self, input, target):
        return torch.sum((input - target) ** 2) / input.data.nelement()
    def forward(self, output, label):
        self.perceptualLoss = self.mse_loss(self.loss_network(output),self.loss_network(label))
        return self.perceptualLoss

def import_model(model_input, in_channel=3, out_channel=1):
    model_test = eval(model_input)(in_channel, out_channel)
    return model_test

def setting_cuda(gpus, model):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        gpu_list = gpus if type(gpus) is list else gpus.split(",")
        model.cuda()
        if len(gpu_list) > 1:
            model = torch.nn.DataParallel(model)

        print("Use gpu:", gpu_list, "to train.")
    else:
        gpu_list = []

    return gpu_list, model