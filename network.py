import torch
import torch.nn as nn
import torch.nn.functional as f


class AlphaZeroNet(nn.Module):

    def __init__(self):
        super(AlphaZeroNet, self).__init__()

        #first layer to reach the board
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1)

        #add a second convolutional layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        #define the policy head to play 
        self.policy_conv = nn.Conv2d(64, 2, kernel_size = 1)
        #output is seven since we have 7 different columns in Connected 4
        self.policy_fc = nn.Linear(2*6*7, 7)

        #define the value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size = 1)
        self.value_fc1 = nn.Linear(1*6*7, 64)
        self.value_fc2 = nn.Linear(64, 1)


    def forward(self, x):

        #inject the input data into 2 convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        #compute the policy
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 2*6*7)
        policy = self.policy_fc(p)

        #compute the value
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 1*6*7)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value




