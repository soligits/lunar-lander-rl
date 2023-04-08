import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        hidden = F.relu(self.layer1(state))
        hidden = F.relu(self.layer2(hidden))
        hidden = F.relu(self.layer3(hidden))
        actions = self.layer4(hidden)

        return actions
