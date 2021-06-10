import torch.nn as nn
from torch import cat
import torch.nn.functional as F

class MultiPatch(nn.Module):
    def __init__(self):
        super(MultiPatch, self).__init__()

        self.convnet3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        
        self.convnet1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
            
        self.fc0 = nn.Sequential(
            nn.Linear(84480, 1024),
            nn.ReLU(inplace=True))
        
        self.fc1 = nn.Sequential(
            nn.Linear(22528, 1024),
            nn.ReLU(inplace=True))
        
        self.fc2 = nn.Sequential(
            nn.Linear(17920, 1024),
            nn.ReLU(inplace=True))
        
        self.fc3 = nn.Sequential(
            nn.Linear(10240, 1024),
            nn.ReLU(inplace=True))
        
        self.fc4 = nn.Sequential(
            nn.Linear(39424, 1024),
            nn.ReLU(inplace=True))
        
        self.fc5 = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(inplace=True))
        
        self.fc6 = nn.Sequential(
            nn.Linear(7680, 1024),
            nn.ReLU(inplace=True))
            
        self.fc1_0 = nn.Sequential(
            nn.Linear(78848, 1024),
            nn.ReLU(inplace=True))
        
        self.fc1_1 = nn.Sequential(
            nn.Linear(13824, 1024),
            nn.ReLU(inplace=True))
        
        self.fc1_2 = nn.Sequential(
            nn.Linear(5120, 1024),
            nn.ReLU(inplace=True))
        
        self.fc1_3 = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.ReLU(inplace=True))
        
        self.fc1_4 = nn.Sequential(
            nn.Linear(25600, 1024),
            nn.ReLU(inplace=True))
        
        self.fc1_5 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True))
        
        self.fc1_6 = nn.Sequential(
            nn.Linear(7608, 1024),
            nn.ReLU(inplace=True))

    def forward(self, X):
        patch_1 = X.float()
        patch_1 = patch_1.cuda()
        if X.shape[1] == 1 :
            patch_1 = self.convnet1(patch_1)
        else :
            patch_1 = self.convnet3(patch_1)
        patch_1 = patch_1.view(patch_1.size()[0], -1)
        #print(patch_1.shape)
        
        if patch_1.shape[1]== 84480:
            patch_1 = self.fc0(patch_1)
        elif patch_1.shape[1]== 22528:
            patch_1 = self.fc1(patch_1)
        elif patch_1.shape[1]== 17920:
            patch_1 = self.fc2(patch_1)
        elif patch_1.shape[1]== 10240:
            patch_1 = self.fc3(patch_1)
        elif patch_1.shape[1]== 39424:
            patch_1 = self.fc4(patch_1)
        elif patch_1.shape[1]== 9216:
            patch_1 = self.fc5(patch_1)
        elif patch_1.shape[1]== 7680:
            patch_1 = self.fc6(patch_1)
        elif patch_1.shape[1]== 78848:
            patch_1 = self.fc1_0(patch_1)
        elif patch_1.shape[1]== 13824:
            patch_1 = self.fc1_1(patch_1)
        elif patch_1.shape[1]== 5120:
            patch_1 = self.fc1_2(patch_1)
        elif patch_1.shape[1]== 4608:
            patch_1 = self.fc1_3(patch_1)
        elif patch_1.shape[1]== 25600:
            patch_1 = self.fc1_4(patch_1)
        elif patch_1.shape[1]== 4096:
            patch_1 = self.fc1_5(patch_1)
        elif patch_1.shape[1]== 7608:
            patch_1 = self.fc1_6(patch_1)
        else :
            # patch prÃ©sent en cas d'erreur
            patch_1 = self.fc6(patch_1)

        return patch_1

class SiameseNet(nn.Module):
    def __init__(self, multi_patch, nb_patch):
        super(SiameseNet, self).__init__()
        self.multi_patch0 = multi_patch
        self.nb_patch = nb_patch
        if nb_patch > 1 :
            self.multi_patch1 = multi_patch
        if nb_patch > 2 :
            self.multi_patch2 = multi_patch
        if nb_patch > 3 :
            self.multi_patch3 = multi_patch
        if nb_patch > 4 :
            self.multi_patch4 = multi_patch
        if nb_patch > 5 :
            self.multi_patch5 = multi_patch
        if nb_patch > 6 :
            self.multi_patch6 = multi_patch

    def forward_once(self, X0, X1, multi_patch):
        output0 = multi_patch(X0)
        output1 = multi_patch(X1)
        
        return output0, output1

    def forward(self, X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13):
        if self.nb_patch < 2 :
            return self.forward_once(X0, X1, self.multi_patch0)
        
        elif self.nb_patch < 3 :
            output0_0, output0_1 = self.forward_once(X0, X1, self.multi_patch0)
            output1_0, output1_1 = self.forward_once(X2, X3, self.multi_patch1)
        
            output0 = cat((output0_0, output1_0), dim=1)
            output1 = cat((output0_1, output1_1), dim=1)
            
            return output0, output1
        
        elif self.nb_patch < 4 :
            output0_0, output0_1 = self.forward_once(X0, X1, self.multi_patch0)
            output1_0, output1_1 = self.forward_once(X2, X3, self.multi_patch1)
            output2_0, output2_1 = self.forward_once(X4, X5, self.multi_patch2)
        
            output0 = cat((output0_0, output1_0, output2_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1), dim=1)
            
            return output0, output1
        
        elif self.nb_patch < 5 :
            output0_0, output0_1 = self.forward_once(X0, X1, self.multi_patch0)
            output1_0, output1_1 = self.forward_once(X2, X3, self.multi_patch1)
            output2_0, output2_1 = self.forward_once(X4, X5, self.multi_patch2)
            output3_0, output3_1 = self.forward_once(X6, X7, self.multi_patch3)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1), dim=1)
            
            return output0, output1
        
        elif self.nb_patch < 6 :
            output0_0, output0_1 = self.forward_once(X0, X1, self.multi_patch0)
            output1_0, output1_1 = self.forward_once(X2, X3, self.multi_patch1)
            output2_0, output2_1 = self.forward_once(X4, X5, self.multi_patch2)
            output3_0, output3_1 = self.forward_once(X6, X7, self.multi_patch3)
            output4_0, output4_1 = self.forward_once(X8, X9, self.multi_patch4)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0, output4_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1, output4_1), dim=1)
            
            return output0, output1
        
        elif self.nb_patch < 7 :
            output0_0, output0_1 = self.forward_once(X0, X1, self.multi_patch0)
            output1_0, output1_1 = self.forward_once(X2, X4, self.multi_patch1)
            output2_0, output2_1 = self.forward_once(X4, X5, self.multi_patch2)
            output3_0, output3_1 = self.forward_once(X6, X7, self.multi_patch3)
            output4_0, output4_1 = self.forward_once(X8, X9, self.multi_patch4)
            output5_0, output5_1 = self.forward_once(X10, X11, self.multi_patch5)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0, output4_0, output5_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1, output4_1, output5_1), dim=1)
            
            return output0, output1
        
        elif self.nb_patch < 8 :
            output0_0, output0_1 = self.forward_once(X0, X1, self.multi_patch0)
            output1_0, output1_1 = self.forward_once(X2, X3, self.multi_patch1)
            output2_0, output2_1 = self.forward_once(X4, X5, self.multi_patch2)
            output3_0, output3_1 = self.forward_once(X6, X7, self.multi_patch3)
            output4_0, output4_1 = self.forward_once(X8, X9, self.multi_patch4)
            output5_0, output5_1 = self.forward_once(X10, X11, self.multi_patch5)
            output6_0, output6_1 = self.forward_once(X12, X13, self.multi_patch6)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0, output4_0, output5_0, output6_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1, output4_1, output5_1, output6_1), dim=1)
            
            return output0, output1

class TripletNet(nn.Module):
    def __init__(self, multi_patch, nb_patch):
        super(TripletNet, self).__init__()
        self.multi_patch0 = multi_patch
        self.nb_patch = nb_patch
        if nb_patch > 1 :
            self.multi_patch1 = multi_patch
        if nb_patch > 2 :
            self.multi_patch2 = multi_patch
        if nb_patch > 3 :
            self.multi_patch3 = multi_patch
        if nb_patch > 4 :
            self.multi_patch4 = multi_patch
        if nb_patch > 5 :
            self.multi_patch5 = multi_patch
        if nb_patch > 6 :
            self.multi_patch6 = multi_patch
        
    def forward_once(self, X0, X1, X2, multi_patch):
        output0 = multi_patch(X0)
        output1 = multi_patch(X1)
        output2 = multi_patch(X2)
        
        return output0, output1, output2

    def forward(self, X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20):
        if self.nb_patch < 2 :
            return self.forward_once(X0, X1, X2, self.multi_patch0)
        
        elif self.nb_patch < 3 :
            output0_0, output0_1, output0_2 = self.forward_once(X0, X1, X2, self.multi_patch0)
            output1_0, output1_1, output1_2 = self.forward_once(X3, X4, X5, self.multi_patch1)
        
            output0 = cat((output0_0, output1_0), dim=1)
            output1 = cat((output0_1, output1_1), dim=1)
            output2 = cat((output0_2, output1_2), dim=1)
            
            return output0, output1, output2
        
        elif self.nb_patch < 4 :
            output0_0, output0_1, output0_2 = self.forward_once(X0, X1, X2, self.multi_patch0)
            output1_0, output1_1, output1_2 = self.forward_once(X3, X4, X5, self.multi_patch1)
            output2_0, output2_1, output2_2 = self.forward_once(X6, X7, X8, self.multi_patch2)
        
            output0 = cat((output0_0, output1_0, output2_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1), dim=1)
            output2 = cat((output0_2, output1_2, output2_2), dim=1)
            
            return output0, output1, output2
        
        elif self.nb_patch < 5 :
            output0_0, output0_1, output0_2 = self.forward_once(X0, X1, X2, self.multi_patch0)
            output1_0, output1_1, output1_2 = self.forward_once(X3, X4, X5, self.multi_patch1)
            output2_0, output2_1, output2_2 = self.forward_once(X6, X7, X8, self.multi_patch2)
            output3_0, output3_1, output3_2 = self.forward_once(X9, X10, X11, self.multi_patch3)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1), dim=1)
            output2 = cat((output0_2, output1_2, output2_2, output3_2), dim=1)
            
            return output0, output1, output2
        
        elif self.nb_patch < 6 :
            output0_0, output0_1, output0_2 = self.forward_once(X0, X1, X2, self.multi_patch0)
            output1_0, output1_1, output1_2 = self.forward_once(X3, X4, X5, self.multi_patch1)
            output2_0, output2_1, output2_2 = self.forward_once(X6, X7, X8, self.multi_patch2)
            output3_0, output3_1, output3_2 = self.forward_once(X9, X10, X11, self.multi_patch3)
            output4_0, output4_1, output4_2 = self.forward_once(X12, X13, X14, self.multi_patch4)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0, output4_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1, output4_1), dim=1)
            output2 = cat((output0_2, output1_2, output2_2, output3_2, output4_2), dim=1)
            
            return output0, output1, output2
        
        elif self.nb_patch < 7 :
            output0_0, output0_1, output0_2 = self.forward_once(X0, X1, X2, self.multi_patch0)
            output1_0, output1_1, output1_2 = self.forward_once(X3, X4, X5, self.multi_patch1)
            output2_0, output2_1, output2_2 = self.forward_once(X6, X7, X8, self.multi_patch2)
            output3_0, output3_1, output3_2 = self.forward_once(X9, X10, X11, self.multi_patch3)
            output4_0, output4_1, output4_2 = self.forward_once(X12, X13, X14, self.multi_patch4)
            output5_0, output5_1, output5_2 = self.forward_once(X15, X16, X17, self.multi_patch5)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0, output4_0, output5_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1, output4_1, output5_1), dim=1)
            output2 = cat((output0_2, output1_2, output2_2, output3_2, output4_2, output5_2), dim=1)
            
            return output0, output1, output2
        
        elif self.nb_patch < 8 :
            output0_0, output0_1, output0_2 = self.forward_once(X0, X1, X2, self.multi_patch0)
            output1_0, output1_1, output1_2 = self.forward_once(X3, X4, X5, self.multi_patch1)
            output2_0, output2_1, output2_2 = self.forward_once(X6, X7, X8, self.multi_patch2)
            output3_0, output3_1, output3_2 = self.forward_once(X9, X10, X11, self.multi_patch3)
            output4_0, output4_1, output4_2 = self.forward_once(X12, X13, X14, self.multi_patch4)
            output5_0, output5_1, output5_2 = self.forward_once(X15, X16, X17, self.multi_patch5)
            output6_0, output6_1, output6_2 = self.forward_once(X18, X19, X20, self.multi_patch6)
        
            output0 = cat((output0_0, output1_0, output2_0, output3_0, output4_0, output5_0, output6_0), dim=1)
            output1 = cat((output0_1, output1_1, output2_1, output3_1, output4_1, output5_1, output6_1), dim=1)
            output2 = cat((output0_2, output1_2, output2_2, output3_2, output4_2, output5_2, output6_2), dim=1)
            
            return output0, output1, output2