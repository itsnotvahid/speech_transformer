from torch import nn


class Block(nn.Module):
    
    def __init__(self, inp, out):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out, out, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.batch_n_1 = nn.BatchNorm2d(out)
        self.batch_n_2 = nn.BatchNorm2d(out)
    
    def forward(self, x):  
        x = self.relu(self.batch_n_1(self.conv1(x)))
        x = self.relu(self.batch_n_2(self.conv2(x)))
        x = self.max_pool(x)
        return x


class CNNFeatureExtractor(nn.Module):

    def __init__(self, layers):
        super().__init__()
        channels = (1, 64, 128)
        self.blocks = nn.ModuleList()

        for i in range(layers):
            if i <= 1:    
                self.blocks.append(Block(channels[i], channels[i+1]))
            else:
                self.blocks.append(Block(channels[2], channels[2]))

    def forward(self, x):
        for module in self.blocks:
            x = module(x)
        batch_size, channels, freq, sequence = x.size()
        x = x.view(batch_size, sequence, channels * freq)
        return x
