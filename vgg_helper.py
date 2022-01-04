import torch 
from torchvision import models 
import torch.nn as nn 
import torch.nn.functional as F

class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers=[2,4,14,21]):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features 

        self.slice0=torch.nn.Sequential()
        self.slice1=torch.nn.Sequential()
        self.slice2=torch.nn.Sequential()
        self.slice3=torch.nn.Sequential()

        for x in range(n_layers[0]):#relu1_1
            self.slice0.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[0],n_layers[1]): #relu1_2
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[1],n_layers[2]): #relu3_2
            self.slice2.add_module(str(x),vgg_pretrained_features[x])

        for x in range(n_layers[2],n_layers[3]):#relu4_2
            self.slice3.add_module(str(x),vgg_pretrained_features[x])

        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False
        

    
    def forward(self,x):
        h0=self.slice0(x)
        h1=self.slice1(h0)
        h2=self.slice2(h1)
        h3=self.slice3(h2)

        return h0,h1,h2,h3


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device) #conv1_1,conv1_2,conv2_2,conv3_3
MSE_Loss=nn.MSELoss(reduction="mean")

def caluclate_contentloss(synth_img,img_p): #W_l

     real_0,real_1,real_2,real_3=perceptual_net(img_p)
     synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_img)

     perceptual_loss=0


     perceptual_loss+=MSE_Loss(synth_0,real_0)
     perceptual_loss+=MSE_Loss(synth_1,real_1)

     perceptual_loss+=MSE_Loss(synth_2,real_2)
     perceptual_loss+=MSE_Loss(synth_3,real_3)



     return perceptual_loss



class StyleLoss(nn.Module):
     def __init__(self, target_feature):
          super(StyleLoss, self).__init__()
          self.target = self.gram_matrix(target_feature).detach()
     def forward(self, input):
          G = self.gram_matrix(input)
          self.loss = F.mse_loss(G, self.target)
          return self.loss
     def gram_matrix(self,input):
          a, b, c, d = input.size()  
          features = input.view(a * b, c * d)  

          G = torch.mm(features, features.t())  
          return G.div(a * b * c * d)




def caluclate_styleloss(synth_img,img_p):

     _,_,_,style_real=perceptual_net(img_p) #conv3_3
     _,_,_,style_synth=perceptual_net(synth_img)

     style_loss=StyleLoss(style_real)

     loss=style_loss(style_synth)

     return loss

 


