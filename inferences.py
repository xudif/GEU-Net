from data_loader import Test_Dataset
from torchvision import transforms
from utils.utils import parse_args, import_model, setting_cuda
from torch.utils.data import DataLoader
import torch
import os
import collections
import numpy as np
import scipy
import fusion_morphology

version = "v3"
type = "maskEdge"

parser = parse_args()
# Preprocess and load data
transformations = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.44], [0.26])])
test_set = Test_Dataset(transforms=transformations, path_test=parser.test)

test_loader = DataLoader(test_set, batch_size=parser.batchsize_test, shuffle=False)

# load pre-trained _model
os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpus

net = import_model(parser.model, parser.model_inchannel, parser.model_outchannel)

try:
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
except Exception as e:
    print('load model error')
    print(e)

# setting cuda if needed
gpus, net = setting_cuda(parser.gpus, net)

is_cuda = len(gpus) >= 1

net.eval()

for i, img_test in enumerate(test_loader):
    if is_cuda:
        img_test = img_test.cuda()
    output = net(img_test)
    output = output.cpu().detach().numpy()
    result = output.astype(np.float32) * 255.
    result = np.clip(result, 0, 255).astype('uint8')
    confidence_map = np.argmax(result, axis=1)
    confidence_map = confidence_map.squeeze()
    img_path = os.path.join('./image/test_pred/' + version + '/' + type + '/initial_decisionmap/', "%02d.png" % (i+1))
    scipy.misc.imsave(img_path, confidence_map)

fusion_morphology.M_fusion(version, type)

print('Finished testing')

