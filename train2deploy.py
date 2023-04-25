from models.repvgg import repvgg_model_convert
import torch
train_path= '/home/guo/RepVgg_Demo/checkpoints/emo/best.pth'
train_model=torch.load(train_path)
deploy_model = repvgg_model_convert(train_model, save_path='repvgg_deploy_emo.pth')