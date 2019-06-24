import torch.nn as nn
import torch
import torch.nn.functional as F
import os

from model.resnet import BasicBlock
from model.faceAlexnet import AgeAlexNet
from utils.network import Conv2d #same padding
from utils.io import Img_to_zero_center

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = Conv2d(69, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.conv3 = Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001, track_running_stats=True)
        self.conv4 = Conv2d(256, 512, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(512, eps=0.001, track_running_stats=True)
        self.conv5 = Conv2d(512, 512, kernel_size=4, stride=2)

    def forward(self, x,condition):
        x = self.lrelu(self.conv1(x))
        x=torch.cat((x,condition),1)
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = Conv2d(32, 3, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.bn4 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn5 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.repeat_blocks=self._make_repeat_blocks(BasicBlock(128,128),6)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0,output_padding=1)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def _make_repeat_blocks(self,block,repeat_times):
        layers=[]
        for i in range(repeat_times):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x,condition=None):
        if condition is not None:
            x=torch.cat((x,condition),1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.repeat_blocks(x)
        x = self.deconv1(x)
        x = self.relu(self.bn4(x))
        x = self.deconv2(x)
        x = self.relu(self.bn5(x))
        x = self.tanh(self.conv4(x))
        return x

class IPCGANs:
    def __init__(self,lr=0.01,age_classifier_path=None,gan_loss_weight=75,feature_loss_weight=0.5e-4,age_loss_weight=30):

        self.d_lr=lr
        self.g_lr=lr

        self.generator=Generator().cuda()
        self.discriminator=PatchDiscriminator().cuda()
        if age_classifier_path is not None:
            self.age_classifier=AgeAlexNet(pretrainded=True,modelpath=age_classifier_path).cuda()
        else:
            self.age_classifier = AgeAlexNet(pretrainded=False).cuda()
        self.MSEloss=nn.MSELoss().cuda()
        self.CrossEntropyLoss=nn.CrossEntropyLoss().cuda()

        self.gan_loss_weight=gan_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.age_loss_weight=age_loss_weight

        self.d_optim = torch.optim.Adam(self.discriminator.parameters(),self.d_lr,betas=(0.5,0.99))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=(0.5, 0.99))

    def save_model(self,dir,filename):
        torch.save(self.generator.state_dict(),os.path.join(dir,"g"+filename))
        torch.save(self.discriminator.state_dict(),os.path.join(dir,"d"+filename))

    def load_generator_state_dict(self,state_dict):
        pretrained_dict = state_dict
        # step2: get model state_dict
        model_dict = self.generator.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.generator.load_state_dict(model_dict)

    def test_generate(self,source_img_128,condition):
        self.generator.eval()
        with torch.no_grad():
            generate_image=self.generator(source_img_128,condition)
        return generate_image

    def cuda(self):
        self.generator=self.generator.cuda()

    def train(self,source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,\
               fake_label_64, age_label):
        '''

        :param source_img_227: use this img to extract conv5 feature
        :param source_img_128: use this img to generate face in target age
        :param true_label_img:
        :param true_label_128:
        :param true_label_64:
        :param fake_label_64:
        :param age_label:
        :return:
        '''

        ###################################gan_loss###############################
        self.g_source=self.generator(source_img_128,condition=true_label_128)

        #real img, right age label
        #logit means prob which hasn't been normalized

        #d1 logit ,discriminator 1 means true,0 means false.
        d1_logit=self.discriminator(true_label_img,condition=true_label_64)

        d1_real_loss=self.MSEloss(d1_logit,torch.ones((d1_logit.size())).cuda())

        #real img, false label
        d2_logit=self.discriminator(true_label_img,condition=fake_label_64)
        d2_fake_loss=self.MSEloss(d2_logit,torch.zeros((d1_logit.size())).cuda())

        #fake img,real label
        d3_logit=self.discriminator(self.g_source,condition=true_label_64)
        d3_fake_loss=self.MSEloss(d3_logit,torch.zeros((d1_logit.size())).cuda())#use this for discriminator
        d3_real_loss=self.MSEloss(d3_logit,torch.ones((d1_logit.size())).cuda())#use this for genrator

        self.d_loss=(1./2 * (d1_real_loss + 1. / 2 * (d2_fake_loss + d3_fake_loss))) * self.gan_loss_weight
        g_loss=(1./2*d3_real_loss)*self.gan_loss_weight


        ################################feature_loss#############################

        self.age_classifier(source_img_227)
        source_feature=self.age_classifier.conv5_feature

        generate_img_227 = F.interpolate(self.g_source, (227, 227), mode="bilinear", align_corners=True)
        generate_img_227 = Img_to_zero_center()(generate_img_227)

        self.age_classifier(generate_img_227)
        generate_feature =self.age_classifier.conv5_feature
        self.feature_loss=self.MSEloss(source_feature,generate_feature)

        ################################age_cls_loss##############################



        age_logit=self.age_classifier(generate_img_227)
        self.age_loss=self.CrossEntropyLoss(age_logit,age_label)

        self.g_loss=self.age_loss+g_loss+self.feature_loss


if __name__=="__main__":
    tensor=torch.ones((2,3,128,128)).cuda()
    condition=torch.ones((2,5,64,64)).cuda()

    discriminator=PatchDiscriminator().cuda()
    from tensorboardX import SummaryWriter
    print(discriminator(tensor,condition).size())
    with SummaryWriter(comment='Net1')as w:
        w.add_graph(discriminator, (tensor,condition))
    dsa
