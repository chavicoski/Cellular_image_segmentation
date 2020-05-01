import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_, dirac_, constant_

#####################
# Auxiliary modules #
#####################

class Conv_block(nn.Module):
    '''
    Basic convolutional block of the U-net
        (Conv + BN + ReLU) * 2
    ''' 
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Conv_block, self).__init__()
        # Define the block layers
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Create the block module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Down_block(nn.Module):
    '''
    Downsamplig step of U-net
        max-pool + Conv_block
    '''
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Down_block, self).__init__()
        # Define the block architecture
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_block(in_channels, out_channels, batch_norm=batch_norm)
        )

    def forward(self, x):
        return self.model(x)


class Up_block(nn.Module):
    '''
    Upsampling step of U-net
        Upsampling_layer + Conv_block
    '''
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Up_block, self).__init__()
        # Create block layers
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv_block = Conv_block(in_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x, x_skip):
        x = self.conv_t(x)
        # Concat upsample output with skip connection before conv_block
        x_cat = torch.cat([x, x_skip], dim=1)
        return self.conv_block(x_cat)


class Out_block(nn.Module):
    '''
    Last convolutional block of the net
    '''
    def __init__(self, in_channels, out_channels):
        super(Out_block, self).__init__()
        # Create block layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if out_channels == 1:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        return self.act(x)

#########
# U-net #
#########

class U_net(nn.Module):
    '''
    Module that implements the final U-net model
    '''
    def __init__(self, in_channels=3, out_channels=1, batch_norm=True, dropout=0.5):
        super(U_net, self).__init__()

        # Define the model layers
        self.in_block = Conv_block(in_channels, 64)
        # Downsampling part
        self.down1 = Down_block(64, 128, batch_norm=batch_norm)
        self.down2 = Down_block(128, 256, batch_norm=batch_norm)
        self.down3 = Down_block(256, 512, batch_norm=batch_norm)
        self.down4 = Down_block(512, 1024, batch_norm=batch_norm)
        # Data augmentation dropout
        self.dropout = dropout > 0.0
        if self.dropout:
            self.drop = nn.Dropout(p=dropout)
        # Upsampling part
        self.up1 = Up_block(1024, 512, batch_norm=batch_norm)
        self.up2 = Up_block(512, 256, batch_norm=batch_norm)
        self.up3 = Up_block(256, 128, batch_norm=batch_norm)
        self.up4 = Up_block(128, 64, batch_norm=batch_norm)
        self.out_block = Out_block(64, out_channels)

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.dropout:
            x5 = self.drop(x5)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        out = self.out_block(x9)
        return out

    def get_criterion(self):
        # This loss combines sigmoid layer with BCELoss for better numerical stability
        return nn.BCEWithLogitsLoss()

    def get_optimizer(self, opt="Adam", lr=0.0002):
        if opt == "Adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "SGD":
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def init_weights(self, init_func_name="he_normal"):

        # Select the initializer
        if init_func_name == "dirac":
            initializer = dirac_
        elif init_func_name == "xavier_uniform":
            initializer = xavier_uniform_
        elif init_func_name == "xavier_normal":
            initializer = xavier_normal_
        elif init_func_name == "he_normal":
            return  # This is the default initializer for Conv layers
        else:
            print(f"The initializer {init_func_name} is not valid!")
            return

        # Create auxiliary function to initialize each module
        def weight_initializer(module):
            if isinstance(module, nn.Conv2d):
                initializer(module.weight)
                constant_(module.bias, 0)

        # Apply the auxiliary function to each submodule
        self.apply(weight_initializer)




        



