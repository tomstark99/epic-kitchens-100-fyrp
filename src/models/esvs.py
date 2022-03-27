import torch as t
import torch.nn as nn
import torch.nn.functional as F

class MTRN(nn.Module):
    
    def __init__(self, frame_count: int):
        super().__init__()
        self.frame_count = frame_count
        self.fc1 = nn.Linear(256 * frame_count, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 397)
    
    def forward(self, x):
        x = x.view(-1, 256 * self.frame_count)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3_verb(x)
        
        return x

class V_MTRN(nn.Module):
    
    def __init__(self, frame_count: int, hidden_layer_size: int, dropout_count: int, dropout_probability: int = 0.5):
        super().__init__()
        if dropout_probability < 0 or dropout_probability > 1:
            raise ValueError(f'Probability needs to be between 0 and 1, was: {dropout_probability}')
        self.frame_count = frame_count
        self.dropout_count = dropout_count
        self.fc1 = nn.Linear(256 * frame_count, hidden_layer_size)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.fc2 = nn.Linear(hidden_layer_size, 512)
        self.fc3_verb = nn.Linear(512, 97)
    
    def forward(self, x):
        x = x.view(-1, 256 * self.frame_count)
        x = F.relu(self.fc1(x))
        if self.dropout_count >= 1:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout_count == 2:
            x = self.dropout(x)
        x = self.fc3_verb(x)
        
        return x

class N_MTRN(nn.Module):
    
    def __init__(self, frame_count: int, hidden_layer_size: int, dropout_count: int, dropout_probability: int = 0.5):
        super().__init__()
        if dropout_probability < 0 or dropout_probability > 1:
            raise ValueError(f'Probability needs to be between 0 and 1, was: {dropout_probability}')
        self.frame_count = frame_count
        self.dropout_count = dropout_count
        self.fc1 = nn.Linear(256 * frame_count, hidden_layer_size)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.fc2 = nn.Linear(hidden_layer_size, 512)
        self.fc3_noun = nn.Linear(512, 300)
    
    def forward(self, x):
        x = x.view(-1, 256 * self.frame_count)
        x = F.relu(self.fc1(x))
        if self.dropout_count >= 1:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout_count == 2:
            x = self.dropout(x)
        x = self.fc3_noun(x)
        
        return x
