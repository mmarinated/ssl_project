
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

from modelzoo import loss_function
from modelzoo import *
from datetime import datetime

from helper import collate_fn, compute_ts_road_map

class Learner:
    def __init__(self, *,
            model_tag,
            device,
            resnet_style='18',
            weights='random',
            model_type='ae',
            checkpoint_folder='/scratch/sc6957/dlproject/checkpoints/',
            tensorboard_log_dir='/scratch/sc6957/dlproject/tb_logs'):
        
        self.device = device

        self.pretrained = {
            'random' : False,
            'imagenet' : True,
            'ssl'    : False
        }[weights]

        self.model_type = model_type
        print('Model Type: {0}'.format(model_type))

        if self.model_type == 'ae':
            self.model = autoencoder(resnet_style=resnet_style, pretrained=self.pretrained)
        elif self.model_type == 'vae':
            self.model = vae(resnet_style=resnet_style, pretrained=self.pretrained)

        sample_input = torch.rand([4,6,3,256,306])
        print('{} Model Summary'.format(model_type))
        self.model.summarize(sample_input)

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)
        self.criterion = nn.BCELoss() # check criterion function

        today = datetime.now().date()
        
        
        self.checkpoint_tag = '{0}-{1}{2}'.format(today.day,today.month, model_tag)
        self.checkpoint_path = checkpoint_folder + 'checkpoint_{0}.pth.tar'.format(self.checkpoint_tag)
        self.best_checkpoint_path = checkpoint_folder + 'checkpoint_{0}_best.pth.tar'.format(self.checkpoint_tag)

        self.epoch = 0
        self.best_loss = np.inf
        self.best_ts = 0

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=model_tag)

    
    # def criterion(self, pred_maps, road_images, mu=None, logvar=None):
    #     """return 
    #         loss, CE, KLD if mu is passed else
    #         loss, nan, nan"""

    #     if mu is None:
    #         loss = self._criterion(pred_maps, road_images.float())
    #         return loss, torch.tensor(float('nan')), torch.tensor(float('nan'))
    #     else:
    #         CE = self._criterion(pred_maps, road_images.float())
    #         KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #         return 0.9*CE + 0.1*KLD, CE, KLD

    def init_loaders(self, train_labeled_set, val_labeled_set, test_labeled_set=None, batch_size=16):
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(train_labeled_set, 
            batch_size=self.batch_size, shuffle=True, drop_last=True,
            num_workers=4, collate_fn=collate_fn)
        self.val_loader   = torch.utils.data.DataLoader(val_labeled_set, 
            batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

        if test_labeled_set is not None:
            self.test_loader = torch.utils.data.DataLoader(test_labeled_set, 
                batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
        else:
            self.test_loader = None

        self.dataloaders = {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}


    def restore_model(self, checkpoint_path):
        print('Restoring checkpoint from {}'.format(checkpoint_path))
        state = torch.load(checkpoint_path)
        self.epoch = state['epoch']
        self.best_loss = state['best_loss']
        self.best_ts = state['best_ts']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])


    def train(self, num_epochs, threshold, tensorboard_log_dir):
        while self.epoch < num_epochs:
            print('Epoch {}/{}'.format(self.epoch, num_epochs - 1))
            print('-' * 10)
            self.epoch += 1

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                threat_score = 0.0

                # Iterate over data.
                for i, temp_batch in enumerate(tqdm(self.dataloaders[phase])):
                    samples, targets, road_images, extras  = temp_batch
                    samples = torch.stack(samples).to(self.device)
                    road_images = torch.stack(road_images).to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.model_type == 'ae':
                            pred_maps = self.model(samples)
                            loss = self.criterion(pred_maps, road_images.float())
                        elif self.model_type == 'vae':
                            pred_maps, mu, logvar = self.model(samples,phase == 'train')
                            loss, CE, KLD = loss_function(pred_maps, road_images, mu, logvar)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()
                        else:
                            for pred_map,road_image in zip(pred_maps,road_images):
                                ts_road_map = compute_ts_road_map(pred_map > threshold, road_image)
                                threat_score += ts_road_map

                    running_loss += loss.item() #*batch_size

                    # tensorboard logging
                    if phase == 'train':
                        writer_idx = self.epoch * len(self.train_loader) + i
                        self.writer.add_scalar(phase+'_loss', loss.item(), writer_idx)
                        if self.model_type == 'vae':
                            self.writer.add_scalar(phase+'_loss_CE', CE.item(), writer_idx)
                            self.writer.add_scalar(phase+'_loss_KLD', KLD.item(), writer_idx)

                pass # end for batch loop

                    # statistics
                if phase == 'train':
                    running_loss = running_loss / len(self.train_loader.dataset) # per batch, per sample
                    print(phase, 'running_loss:', round(running_loss,4))
                    self.writer.add_scalar(phase+'_total_loss', running_loss, self.epoch)
                else:
                    num_test_samples = len(self.val_loader.dataset)
                    running_loss = running_loss / num_test_samples
                    print(phase,'running_loss:', round(running_loss,4), 'cumulative_threat_score:', round(threat_score.item(),4), \
                        'val_len:', num_test_samples, '\n mean_threat_score:',round(threat_score.item() / num_test_samples,4))#, iou / len(val_set))
                    self.writer.add_scalar(phase+'_ts', threat_score.item()/num_test_samples, self.epoch)
                    self.writer.add_scalar(phase+'_total_loss', running_loss, self.epoch)

            self._update_states(running_loss, threat_score)

    def _update_states(self, running_loss, threat_score):

        cur_state = {
            'epoch': self.epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_ts': self.best_ts
        }

        # Saving best loss model so far
        if running_loss < self.best_loss:
            self.best_loss = running_loss
            torch.save(cur_state, self.best_checkpoint_path)
            print('best_loss model after %d epoch saved...' % (self.epoch+1))

        # Saving best ts model so far
        if threat_score > self.best_ts:
            self.best_ts = threat_score
            torch.save(cur_state, self.best_checkpoint_path)
            print('best_ts model after %d epoch saved...' % (self.epoch+1))

        # save model per epoch
        torch.save(cur_state, self.checkpoint_path)
        print('model after %d epoch saved...' % (self.epoch+1))



    def report_loss(self, split_name="val", model_path=None, display_images=True, 
                    batches_idces_to_display=slice(None), threshold = 0.5):
        assert split_name in ["train", "val", "test"]
        
        print('Estimating performance on {} set'.format(split_name))
        
        if model_path is not None:
            print('Restoring Best checkpoint from {}'.format(model_path))
            state = torch.load(model_path)
            # self.restore_model()
            self.model.load_state_dict(state['state_dict'])
        
        self.model.eval()

        threat_score = 0.0
        total = 0.0
        
        batches_idces_to_display = np.arange(len(self.dataloaders[split_name]))[batches_idces_to_display]
        
        for i, temp_batch in enumerate(self.dataloaders[split_name]):
            if i not in batches_idces_to_display:
                continue
            total += len(temp_batch[0])
            samples, targets, road_images, extra = temp_batch
            samples = torch.stack(samples).to(self.device)
            road_images = torch.stack(road_images).to(self.device)
            
            if self.model_type == 'ae':
                pred_maps = self.model(samples)
            elif self.model_type == 'vae':
                pred_maps, mu, logvar = self.model(samples,is_training=False)

            for pred_map, road_image in zip(pred_maps,road_images):
                ts_road_map = compute_ts_road_map(pred_map > threshold, road_image)
                threat_score += ts_road_map

            if display_images:
                for sample, pred_map, road_image in zip(samples,pred_maps,road_images):
                    print('Test Batch: {}'.format(i))
                    self.show_predictions(sample, pred_map, road_image, threshold)
                    # CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT


        threat_score /= total
        print('Total samples: {}, Total Threat Score: {}'.format(total,total*threat_score))
        print('Mean Threat Score is: {}'.format(threat_score))


    def show_predictions(self, sample, pred_map, road_image, threshold):
        plt.imshow(torchvision.utils.make_grid(sample.detach().cpu(), nrow=3).numpy().transpose(1, 2, 0))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Road Map Comparison')
        ax1.imshow(road_image.detach().cpu(), cmap='binary')
        ax1.set_title('Original Road Map')
        ax2.imshow((pred_map > threshold).detach().cpu(), cmap='binary')
        ax2.set_title('Predicted Road Map')
        plt.show()
        print('-'*20)


# def _how_it_should_look(self):
#     for epoch in epochs:

#         train_losses = self.train(train_loader)
#         self.logg_to_tensorobard(train_losses)
        
#         val_losses   = self.test(test_loader)
#         self.update_best_states(val_losses)


# def train(loader):

#     for .. batch ...:
#         fcast = self.predict(batch)
#         loss = self.loss(fcast, target)

#         compute and update grads

# def test(loader):
#     with torch.no_grad():
#         for .. batch ...:

#             fcast = self.predict(batch)

#             compute and update metrics
