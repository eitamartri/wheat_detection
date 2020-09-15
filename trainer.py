import torch
import saves as sv
import numpy as np
import tqdm


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


def train_model(history, config, model, optimizer,lr_scheduler, train_data_loader, valid_data_loader,device):

    loss_hist = Averager()
    val_loss_hist = Averager()
    itr = 1
    for epoch in range(history.epoch, history.epoch + config.epochs):
        loss_hist.reset()

        for images, targets, image_ids in train_data_loader:
            if itr % 100 == 0:
                torch.cuda.empty_cache()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 1 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")
        history.loss.append(loss_hist.value)

        val_loss_hist.reset()
        for images, targets, image_ids in valid_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                valid_loss_dict = model(images, targets)

            valid_losses = sum(loss for loss in valid_loss_dict.values())
            valid_loss_value = valid_losses.item()

            val_loss_hist.send(valid_loss_value)
        history.val_loss.append(val_loss_hist.value)

        print('val loss: {}'.format(val_loss_hist.value))
        if val_loss_hist.value < history.best_val_loss:
            history.best_val_loss = val_loss_hist.value
            history.epoch = epoch + 1
            print('New best loss: {}'.format(val_loss_hist.value))
            sv.update_model(model,optimizer, history)
            # torch.save(model.state_dict(), 'results/fasterrcnn_resnet50_fpn.pth')
            state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history}
            torch.save(state, 'results/history_' + str(history.model_num) + '.pth.tar')
            

