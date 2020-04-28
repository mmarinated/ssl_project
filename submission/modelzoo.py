import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision
from torchvision import models, transforms

class encoder(nn.Module):
    
    def __init__(self, resnet_style='18', pretrained=False):
        super(encoder, self).__init__()
        self.resnet_style = resnet_style
        self.pretrained = pretrained
        if self.resnet_style == '18' and self.pretrained==False:
            resnet = models.resnet18(pretrained=False)
            
        elif self.resnet_style == '18' and self.pretrained==True:
            resnet = models.resnet18(pretrained=True)
            
        if self.resnet_style == '50' and self.pretrained==False:
            resnet = models.resnet50(pretrained=False)
            
        if self.resnet_style == '50' and self.pretrained==True:
            resnet = models.resnet50(pretrained=True)
            
        resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        
        if self.resnet_style == '18':
            self.resnet_encoder = nn.Sequential(*list(resnet.children())[:-1])
        elif self.resnet_style == '50':
            self.resnet_encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Conv2d(2048, 512, 1))

    def forward(self, x):
        x = self.resnet_encoder(x)
        return x
    
    def summarize(self,x,offset=''):
        print(offset+'Class: {}'.format(type(self).__name__))
        print(offset+'resnet_style: {}, pretrained: {}'.format(self.resnet_style,self.pretrained))
        print(offset+'Passed Input Size:{}'.format(x.shape))
        print(offset+'Output Size:{}'.format(self.forward(x).shape))
    
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        nz = 3072
        ngf = 64
        nc = 1
        self.deconv_decoder = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 3, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf*8, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf*4, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf*2, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.deconv_decoder(x).squeeze(1)
        return x
              
    def summarize(self,x,offset=''):
        print(offset+'Class: {}'.format(type(self).__name__))
        print(offset+'Passed Input Size:{}'.format(x.shape))
        print(offset+'Output Size:{}'.format(self.forward(x).shape))
    
class autoencoder(nn.Module):

    def __init__(self, resnet_style='18', pretrained=False):
        super(autoencoder, self).__init__()
        self.nz = 3072
        self.encoder = encoder(resnet_style=resnet_style, pretrained=pretrained)
        self.decoder = decoder()

    def forward(self, x):
        n_img = x.shape[1]
        z_arr = []
        for i in range(n_img): 
            z_arr.append(self.encoder(x[:,i])) ### Order might matter.
        z = torch.cat(z_arr,1).view(-1,self.nz,8,8)
        pred = self.decoder(z)
        return pred
              
    def summarize(self,x,offset=''):
        print(offset+'Class: {}'.format(type(self).__name__))
        print(offset+'Passed Input Size:{}'.format(x.shape))
        n_img = x.shape[1]
        z_arr = []
        for i in range(n_img): 
            z_arr.append(self.encoder(x[:,i])) ### Order might matter.
        z = torch.cat(z_arr,1).view(-1,self.nz,8,8)
        pred_map = self.decoder(z)
        print(offset+'----')
        self.encoder.summarize(x[:,0],offset=' '*5)
        print(offset+'----')
        print(offset+'Number of encoded states: {}, each of size: {}'.format(len(z_arr),z_arr[0].shape))
        print(offset+'Concatenated Hidden State size: {}'.format(z.shape))
        print(offset+'----')
        self.decoder.summarize(z,offset=' '*5)
        print(offset+'----')
        print(offset+'Output Size:{}'.format(pred_map.shape))
        
class encoder_after_resnet(nn.Module):

    def __init__(self):
        super(encoder_after_resnet, self).__init__()
        

        self.conv = nn.Sequential(
            nn.Conv2d(3072, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4096*2)
        mu = x[:,:4096]
        logvar = x[:,4096:]
        return mu, logvar
              
    def summarize(self,x,offset=''):
        print(offset+'Class: {}'.format(type(self).__name__))
        print(offset+'Passed Input Size:{}'.format(x.shape))
        x = self.conv(x)
        print(offset+'Convolved Encoded state shape: {}'.format(x.shape))
        x = x.view(-1, 4096*2)
        mu = x[:,:4096]
        logvar = x[:,4096:]
        print(offset+'Output Mean Size:{}'.format(mu.shape))
        print(offset+'Output Var Size:{}'.format(logvar.shape))
    
class vae_decoder(nn.Module):
    def __init__(self):
        super(vae_decoder, self).__init__()
        nz = 64
        ngf = 64
        nc = 1
        self.deconv_decoder = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 3, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf*8, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf*4, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf*2, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 64, 8, 8)
        x = self.deconv_decoder(x).squeeze(1)
        return x

    def summarize(self,x,offset=''):
        print(offset+'Class: {}'.format(type(self).__name__))
        print(offset+'Passed Input Size:{}'.format(x.shape))
        x = x.view(-1, 64, 8, 8)
        print(offset+'Input recast into shape: {}'.format(x.shape))
        x = self.deconv_decoder(x).squeeze(1)
        print(offset+'Output Size:{}'.format(x.shape))
              
class vae(nn.Module):

    def __init__(self, resnet_style='18', pretrained=False):
        super(vae, self).__init__()
        
        self.encoder = encoder(resnet_style=resnet_style, pretrained=pretrained)
        self.encoder_after_resnet = encoder_after_resnet()
        self.vae_decoder = vae_decoder()

    def reparameterize(self, is_training, mu, logvar):
        if is_training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, is_training, defined_mu=None):
        n_img = x.shape[1]
        z_arr = []
        for i in range(n_img): 
            z_arr.append(self.encoder(x[:,i]))
        x = torch.cat(z_arr,1)
        
        mu, logvar = self.encoder_after_resnet(x)
        z = self.reparameterize(is_training, mu, logvar)
        if defined_mu is not None:
            z = defined_mu
        pred_map = self.vae_decoder(z)
        return pred_map, mu, logvar
              
    def summarize(self,x,offset=''):
        original_inp_slice = x[:,0]
        print(offset+'Class: {}'.format(type(self).__name__))
        print(offset+'Passed Input Size:{}'.format(x.shape))
        n_img = x.shape[1]
        z_arr = []
        for i in range(n_img): 
            z_arr.append(self.encoder(x[:,i]))
        x = torch.cat(z_arr,1)
        mu, logvar = self.encoder_after_resnet(x)
        z = self.reparameterize(is_training=False,mu= mu,logvar= logvar)
        pred_map = self.vae_decoder(z)
        print(offset+'----')
        self.encoder.summarize(original_inp_slice,offset=' '*5)
        print(offset+'----')
        print(offset+'Number of encoded states: {}, each of size: {}'.format(len(z_arr),z_arr[0].shape))
        print(offset+'Concatenated encoded states shape: {}'.format(x.shape))
        print(offset+'----')
        self.encoder_after_resnet.summarize(x,offset=' '*5)
        print(offset+'----')
        print(offset+'Output Mean Size:{}'.format(mu.shape))
        print(offset+'Output Var Size:{}'.format(logvar.shape))
        print(offset+'Reparameterized Hidden State size: {}'.format(z.shape))
        print(offset+'----')
        self.vae_decoder.summarize(z,offset=' '*5)
        print(offset+'----')
        print(offset+'Output Size:{}'.format(pred_map.shape))
    
def loss_function(pred_maps, road_images, mu, logvar):
    criterion = nn.BCELoss()
    CE = criterion(pred_maps, road_images.float())
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.9*CE + 0.1*KLD, CE, KLD