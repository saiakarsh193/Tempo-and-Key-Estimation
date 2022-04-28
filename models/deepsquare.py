import torch.nn as nn

class DeepSquare(nn.Module):
    def __init__(self, input_size, output_size, k, p_dropout): # input_size(1, F_k, T_k)
        super(DeepSquare, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k

        self.fe_0 = nn.Sequential(
            nn.Conv2d(input_size[0], (1 << 0)*k, (5, 5), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 0)*k),

            nn.Conv2d((1 << 0)*k, (1 << 0)*k, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 0)*k),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(p_dropout)
        )

        self.fe_1 = nn.Sequential(
            nn.Conv2d((1 << 0)*k, (1 << 1)*k, (5, 5), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 1)*k),

            nn.Conv2d((1 << 1)*k, (1 << 1)*k, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 1)*k),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(p_dropout)
        )

        self.fe_2 = nn.Sequential(
            nn.Conv2d((1 << 1)*k, (1 << 2)*k, (5, 5), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 2)*k),

            nn.Conv2d((1 << 2)*k, (1 << 2)*k, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 2)*k),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(p_dropout)
        )

        self.fe_3 = nn.Sequential(
            nn.Conv2d((1 << 2)*k, (1 << 3)*k, (5, 5), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 3)*k),

            nn.Conv2d((1 << 3)*k, (1 << 3)*k, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d((1 << 3)*k),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(p_dropout)
        )
        
        self.final_layer = nn.Sequential(
            nn.Conv2d((1 << 3)*k, output_size, 1, padding=1),
            nn.ReLU()
        )

        self.globalPooling = lambda x: x.mean([len(x.shape)-1, len(x.shape)-2])
    
    def forward(self, X):
        output = self.fe_0(X)
        output = self.fe_1(output)
        output = self.fe_2(output)
        output = self.fe_3(output)
        output = self.final_layer(output)
        output = self.globalPooling(output)
        return output