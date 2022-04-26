import torch.nn as nn

class ShallowTempo(nn.Module):
    def __init__(self, input_size, output_size, k, p_dropout): # (1, F_t, T_t)
        super(ShallowTempo, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.dropout = nn.Dropout(p_dropout)
        self.layer1 = nn.Conv2d(input_size[0], k, (1, 3), padding=1)

        self.poolingLayer = nn.AvgPool2d((input_size[1], 1))
        
        self.layer2 = nn.Conv2d(k, 64*k, (1, input_size[2]), padding=1)
        
        self.layer3 = nn.Conv2d(64*k, output_size, 1, padding=1) # (n, output_size, h', w')
        self.globalPooling = lambda x: x.mean([2, 3]) # (n, output_size, )
        self.relu = nn.ReLU()
    
    def forward(self, X): # X - (batch, C, H, W)
        output = self.relu(self.layer1(X)) # (n, k, h, w)
        output = self.dropout(output)

        output = self.poolingLayer(output) # (n, k, h', w')
        
        output = self.relu(self.layer2(output)) # (n, 64k, h', w')
        output = self.dropout(output)
        output = self.relu(self.layer3(output)) # (n, output_size, h', w')
        output = self.globalPooling(output) # (n, output_size)
        
        return output