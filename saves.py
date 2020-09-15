import os
import csv
import pandas as pd
import torch


def get_model_num(config):
    new_df = False
    if not os.path.exists('results/history.csv'):
        with open('results/history.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['backbone', 'num of pics', 'train/val ratio', 'anchor sizes', 'anchor ratios',
                                 'epochs','batch_size','learning_rate','betas','step_size',
                                 'gamma','weight_decay','best loss', 'model number'])                                                #TODO add cols with new argumants
            new_df = True

    df = pd.read_csv('results/history.csv')

    model_df = df[df['backbone'] == config.backbone][df['num of pics'] == config.num_of_pics_to_use]\
    [df['train/val ratio'] == config.train_ratio][df['anchor sizes'] == str(config.anchor_sizes)]\
    [df['anchor ratios'] == str(config.aspect_ratios)][df['batch_size'] == config.batch_size] \
    [df['learning_rate'] == config.learning_rate][df['betas'] == str(config.betas)] \
    [df['step_size'] == config.step_size][df['gamma'] == config.gamma][df['weight_decay'] == config.weight_decay]


    if model_df.shape[0] == 1:
        model_number = df.iloc[-1]['model number']
    elif model_df.shape[0] == 0:
        print('Building new model')
        model_number = 0 if new_df else df.iloc[-1]['model number'] + 1
        df.loc[-1]= [config.backbone, config.num_of_pics_to_use, config.train_ratio, config.anchor_sizes,
                     config.aspect_ratios, 0, config.batch_size, config.learning_rate, config.betas, config.step_size,
                     config.gamma, config.weight_decay, 999, model_number]
        df.to_csv('results/history.csv', index=False)
    else:
        print('There is more then one model with the same parameters')
        exit(1)
    return model_number

def update_model(model,optimizer, history):
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history}
    torch.save(state, 'results/history_' + str(history.model_num) + '.pth.tar')

    df = pd.read_csv('results/history.csv')
    filt = (df['model number'] == history.model_num)
    df.loc[filt, 'epochs'] = history.epoch
    df.loc[filt, 'best loss'] = history.best_val_loss

    df.to_csv('results/history.csv', index=False)


