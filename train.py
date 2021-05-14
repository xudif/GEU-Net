from utils.utils import parse_args, import_model, PerceptualLoss, setting_cuda
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from data_loader import TrainDataset, Valid_Dataset
from torch import nn
import os
import random
import scipy.misc
import torch
import numpy as np
import collections

parser = parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set random number seed
setup_seed(66)
# Preprocess and load data
transformations = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.44], [0.26])])

# from data_loader import Dataset
train_set = TrainDataset(parser.train_csv)
valid_set = Valid_Dataset(transforms=transformations, path_val=parser.val)

print("preparing training data ...")
train_loader = DataLoader(train_set, batch_size=parser.batchsize_train, shuffle=True, drop_last=True, num_workers=5)
print("done ...")
print("preparing valid data ...")
valid_loader = DataLoader(valid_set, batch_size=parser.batchsize_valid, shuffle=False, num_workers=3)
print("done ...")

# load pre-trained _model
os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpus

net = import_model(parser.model, parser.model_inchannel, parser.model_outchannel)

if parser.weights is not None:
    pretrained_path = parser.weights
    print("loaded model %s" % pretrained_path)
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[
                target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)
    print("done ...")

# setting cuda if needed
gpus, net = setting_cuda(parser.gpus, net)

is_cuda = len(gpus) >= 1


class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


def train():
    criterion1 = nn.BCEWithLogitsLoss()
    if parser.step2:
        criterion4 = PerceptualLoss(is_cuda)
    if is_cuda:
        criterion1 = criterion1.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=parser.lr, momentum=0.99, weight_decay=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

    avg_loss = 0
    losses = []
    print('training ...')
    for epoch in range(parser.epoch):
        train_loss = Average()
        # scheduler.step(epoch)
        if epoch != 0:
            scheduler.step(avg_loss)
        net.train()

        # freezed the parameters of the BN and IN layers
        def set_norm_eval(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm') != -1:
                m.eval()
            if classname.find('BatchNorm2d') != -1:
                m.eval()

        if parser.step2:
            print('freezing Normalization')
            net.apply(set_norm_eval)
            print('done')

        for index, (img_train, label) in enumerate(train_loader):
            H, W = label.shape[2:]
            label = label.numpy()
            label = torch.LongTensor(label)
            label_one_hot = torch.zeros(parser.batchsize_train, parser.model_inchannel, H, W).scatter_(1, label, 1)

            if is_cuda:
                img_train = img_train.cuda()
                label_one_hot = label_one_hot.cuda()

            optimizer.zero_grad()
            outputs = net(img_train)
            if parser.step2:
                loss1 = criterion1(outputs,
                                   label_one_hot)
                loss2 = criterion4(outputs,
                                   label_one_hot)
                loss = loss1 * parser.w1_bce + loss2 * parser.w2_per
            else:
                loss = criterion1(outputs,
                                  label_one_hot)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())

            if index % 100 == 0:
                losses.append(loss)
                if parser.step2:
                    print("Epoch: [%2d], step: [%2d], loss: [%.8f], BCE_loss: [%.8f], perceptual_loss: [%.8f]" % (
                        (epoch + 1), index, loss, loss1, loss2))
                else:
                    print("Epoch: [%2d], step: [%2d], loss: [%.8f]" % ((epoch + 1), index, loss))
            if index % 200 == 0:
                for i in range(parser.batchsize_train):
                    output = outputs[i, :, :, :].unsqueeze(0)
                    output = output.cpu().detach().numpy()
                    result = output.astype(np.float32) * 255.
                    result = np.clip(result, 0, 255).astype('uint8')
                    confidence_map = np.argmax(result, axis=1)
                    confidence_map = confidence_map.squeeze()
                    label1 = label[i, :, :, :].squeeze()
                    label1 = label1 * 255

                    img_path = os.path.join('./_image/train_pred', "%02d_%02d_pred.png" % (index, i))
                    label_path = os.path.join('./_image/train_pred', "%02d_%02d_label.png" % (index, i))
                    scipy.misc.imsave(img_path, confidence_map)
                    scipy.misc.imsave(label_path, label1)
                torch.save(net.state_dict(), '_model/cp_{}_{}_{}.pth'.format((epoch + 1), index, loss))

                # valid data
                with torch.no_grad():
                    net.eval()
                    for i, img_valid in enumerate(valid_loader):
                        if is_cuda:
                            img_valid = img_valid.cuda()
                        output = net(img_valid)
                        output = output.cpu().detach().numpy()
                        result = output.astype(np.float32) * 255.
                        result = np.clip(result, 0, 255).astype('uint8')
                        confidence_map = np.argmax(result, axis=1)
                        confidence_map = confidence_map.squeeze()
                        img_path = os.path.join('./_image/valid_pred', "%02d_%02d_pred.png" % (epoch, i))
                        scipy.misc.imsave(img_path, confidence_map)
                    net.train()
                    if parser.step2:
                        net.apply(set_norm_eval)
        avg_loss = train_loss.avg
        print("Epoch {}/{}, Loss: {}".format(epoch + 1, parser.epoch, avg_loss))


if __name__ == "__main__":
    train()
