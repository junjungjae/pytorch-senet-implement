import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, out_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        # 논문의 Architecture 기반 SEBlock 구현
        # resnet에 적용할 경우 block 내에서 shortcut을 더해주기 전에 적용
        # 또한, bottleneck block의 특성 상 output channel이 input channel의 4배가 되기 때문에 excitation시 해당 특성 적용
        # self.excitation = nn.Sequential(            
        #     nn.Linear(in_features=out_channels*4, out_features=(out_channels//reduction_ratio)),
        #     nn.ReLU(),
        #     nn.Linear(in_features=(out_channels//reduction_ratio), out_features=out_channels*4),
        #     nn.Sigmoid()
        # )
        
        # 논문의 경우 Fully Connected layer로 excitation이 수행됨
        # FC 대신 Convolution을 이용한 excitation을 적용한 사람이 있어서 마찬가지로 적용해봄
        # 
        self.excitation = nn.Sequential(            
            nn.Conv2d(in_channels=out_channels*4, out_channels=(out_channels//reduction_ratio), padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=(out_channels//reduction_ratio), out_channels=out_channels*4, padding=1, kernel_size=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.squeeze(x)
        out = out.view(out.size(0), out.size(1), 1, 1)  # Global Average Pooling이 적용되었으므로 그에 맞게 reshape
        out = self.excitation(out)
        out = out.view(out.size(0), out.size(1), 1, 1)  # squeeze & Excitation 과정을 모두 거쳤으므로 후속 layer에 맞게 다시 reshape
        return out


class plainblock(nn.Module):
    bottleneck_multiple = 1  # bottleneck block 구현 시 channel을 맞춰주기 위한 계수

    def __init__(self, in_channel, out_channel, stride=1):
        super(plainblock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Convolution 구현부
        # plainblock의 경우 기존 모델들과 유사하게 layer를 쌓음
        # 다만, feature map resizing 과정에서 maxpooling 대신 stride를 2를 부여하여 resizing
        # resizing은 각 Convolution block에서 한번만 이루어짐
        # 연산량 -> maxpooling 승, 학습 측면 -> stride 승(kernel은 trainable 하기 때문에)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)

        self.seblock = SEBlock(out_channel)

        self.shortcut = nn.Sequential()

        # resizing으로 인해 기존 입력 데이터와 차원이 맞지 않거나
        # block의 출력부와 입력값의 차원이 맞지 않는 경우
        # 두 값을 더해주기 위해 shortcut의 차원을 맞추는 과정 수행
        if (stride != 1) or (in_channel != out_channel * plainblock.bottleneck_multiple):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * plainblock.bottleneck_multiple, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * plainblock.bottleneck_multiple)
            )


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.seblock(out) * x  # SEBlock을 통해 추출한 channel별 중요도를 기존 feature에 적용

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class bottleneckblock(nn.Module):
    bottleneck_multiple = 4  # bottleneck block 구현 시 channel을 맞춰주기 위한 계수

    def __init__(self, in_channel, out_channel, stride=1):
        super(bottleneckblock, self).__init__()

        # bottleneck block의 특수한 구조
        # 3*3 kernel을 2개 사용하는 대신 1*1, 3*3, 1*1 kernel 사용
        # 동일한 작업을 수행하는 효과를 얻으면서 연산량 감소 & 활성화 함수 사용 증가로 인한 비선형성 증가의 이점을 가짐
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * bottleneckblock.bottleneck_multiple, kernel_size=(1, 1),
                               stride=(1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * bottleneckblock.bottleneck_multiple)

        self.relu = nn.ReLU(inplace=True)

        self.seblock = SEBlock(out_channel)

        self.shortcut = nn.Sequential()

        # plainblock과 동일하게 resizing이나 입력부와 출력부의 차원이 맞지 않을 경우 맞춰주는 과정 수행
        if (stride != 1) or (in_channel != out_channel * bottleneckblock.bottleneck_multiple):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * bottleneckblock.bottleneck_multiple, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * bottleneckblock.bottleneck_multiple)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.seblock(out) * out  # SEBlock을 통해 추출한 channel별 중요도를 기존 feature에 적용

        out += self.shortcut(x)
        out = self.relu(out)
        return out