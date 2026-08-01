import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):

    def __init__(self, game):
        super(AlphaZeroNet, self).__init__()

        self.rows = game.raws
        self.columns = game.columns
        self.action_size = game.action_size

        #first layer to reach the board
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1)

        #add a second convolutional layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        #define the policy head to play
        self.policy_conv = nn.Conv2d(64, 2, kernel_size = 1)
        #output is seven since we have 7 different columns in Connected 4
        self.policy_fc = nn.Linear(2 * self.rows * self.columns, self.action_size)

        #define the value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size = 1)
        self.value_fc1 = nn.Linear(1 * self.rows * self.columns, 64)
        self.value_fc2 = nn.Linear(64, 1)


    def forward(self, x):

        #accept both (N, rows, cols) and (N, 1, rows, cols): add the channel dim if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)

        #inject the input data into 2 convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        #compute the policy
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 2 * self.rows * self.columns)
        policy = self.policy_fc(p)

        #compute the value
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 1 * self.rows * self.columns)
        v = F.relu(self.value_fc1(v))

        #use tanh to set the value between -1 and +1 either for losing or winning
        value = torch.tanh(self.value_fc2(v))

        return policy, value
