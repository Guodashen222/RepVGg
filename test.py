import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from models.repvgg import create_RepVGG_A0
import torch.nn as nn
classes = ('angry','disgust','fear','happy','neutral','sad','surprise')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.51819474, 0.5250407, 0.4945761], std=[0.24228974, 0.24347611, 0.2530049])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_RepVGG_A0(deploy=True)
num_ftrs = model.linear.in_features
model.linear = nn.Linear(num_ftrs, 7)
model.load_state_dict(torch.load('repvgg_deploy_emo.pth'))

model.eval()
model.to(DEVICE)

path = 'test/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
