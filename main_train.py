import pandas as pd
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
import pickle

# my scripts
import network
import configurations as cfg
import data_handler as dh
import visualaize as vis
import trainer
import saves as sv

#get the run configurations
config = cfg.get_args()

if config.eval_train:
    detection_threshold = 0.5
    # vis.eval_and_plot_on_train(train_data_loader, model,detection_threshold)
    vis.PlotLoss([0, 1, 2, 4], param='learning_rate')
    exit(1)

#choose the relevant device
ChooseDevice=1 # 0=CPU ,1=Check for GPE
if ChooseDevice:
    device = torch.device('cuda:' + str(config.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
else:
    device=torch.device('cpu')
print(device)


#creating the model, optimizer and documentation
model,optimizer,history,lr_scheduler = network.build_model(config,device)
#creating split train and validation datasets
train_data_loader, valid_data_loader = dh.get_data_loaders(config)

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


if __name__ == '__main__':


    trainer.train_model(history, config, model, optimizer,lr_scheduler, train_data_loader, valid_data_loader,device)





