import torch.nn as nn
from se_resnetblock import plainblock, bottleneckblock

class seresnet(nn.Module):
    # 초기값으로 데이터의 클래스 개수, conv block 유형, 각 conv block의 개수를 받음
    # resnet의 경우 block의 개수가 18, 34일 경우 plainblock으로, layer의 개수가 50개 이상의 모델일 경우 bottleneckblock으로 작성
    """
        resnet18 - basic , [2, 2, 2, 2]
        resnet34 - basic , [3, 4, 6, 3]
        resnet50 - bottleneck , [3, 4, 6, 3]
        resnet101 - bottleneck , [3, 4, 23, 3]
        resnet152 - bottleneck , [3, 8, 36, 3]
    """
    def __init__(self, num_classes, blocktype, blocknumlist=None):
        super(seresnet, self).__init__()
        self.in_channel = 64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )

        # 별도 method를 통해 Conv block 생성
        # 2번째 Conv block의 경우 바로 전 단계에서 Maxpooling을 수행하기 때문에 stride resize를 수행하지 않음
        self.conv_2 = self.generateconv('bottleneck', 64, blocknumlist[0], 1)
        self.conv_3 = self.generateconv('bottleneck', 128, blocknumlist[1], 2)
        self.conv_4 = self.generateconv('bottleneck', 256, blocknumlist[2], 2)
        self.conv_5 = self.generateconv('bottleneck', 512, blocknumlist[3], 2)

        # 생성된 모든 feature map에 대해 globalaveragepooling 적용
        # 논문에 자세한 이유는 작성되어 있지 않지만 fully-connected layer와의 차원을 맞추기 용이함과 연산량을 줄이기 위한 목적같음
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # fully-connected layer에서 feature size를 맞춰주기
        if blocktype == 'plain':
            blockmultiple = 1

        elif blocktype == 'bottleneck':
            blockmultiple = 4

        self.fc = nn.Linear(512 * blockmultiple, num_classes)  #

    def generateconv(self, blocktype, in_channel, blocknum, stride):
        blockmodule = []
        stridelist = [stride] + [1] * (blocknum - 1)  # 입력 stride가 1인 경우가 2인 경우 2가지 케이스 고려

        if blocktype == 'plain':
            for stride in stridelist:
                blockmodule.append(plainblock(self.in_channel, in_channel, stride))

        elif blocktype == 'bottleneck':
            for stride in stridelist:
                blockmodule.append(bottleneckblock(self.in_channel, in_channel, stride))
                # Conv block의 첫번째 블록일 경우 resizing 과정을 거치기 때문에 별도 입력값 조정 필요 없음
                # 2번째 블록부터는 전단계의 출력 channel이 4배수가 됐으므로 그에 맞춰줘야함
                self.in_channel = in_channel * 4

        returnsequential = nn.Sequential(*blockmodule)
        return returnsequential

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out