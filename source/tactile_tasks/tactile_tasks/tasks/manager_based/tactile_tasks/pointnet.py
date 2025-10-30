import torch
import torch.nn as nn

class TransformNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, k, N)
        b = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(b, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        ident = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(b, 1)
        x = (x + ident).view(b, self.k, self.k)
        return x

class PointNet(nn.Module):
    def __init__(self, in_channels=3, global_feat_dim=1024, use_feature_transform=True):
        super().__init__()
        self.use_feature_transform = use_feature_transform
        self.tnet1 = TransformNet(k=in_channels)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.tnet2 = TransformNet(k=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, global_feat_dim, 1)
        self.bn3 = nn.BatchNorm1d(global_feat_dim)
        self.relu = nn.ReLU()
        self.global_feat_dim = global_feat_dim

    def forward(self, x):
        # x: (B, C, N)
        trans1 = self.tnet1(x)                       # (B, C, C)
        x = torch.bmm(trans1, x)                     # (B, C, N)
        x = self.relu(self.bn1(self.conv1(x)))       # (B, 64, N)

        if self.use_feature_transform:
            trans2 = self.tnet2(x)                   # (B, 64, 64)
            x = torch.bmm(trans2, x)                 # (B, 64, N)

        x = self.relu(self.bn2(self.conv2(x)))       # (B, 128, N)
        x = self.bn3(self.conv3(x))                  # (B, G, N)
        x = torch.max(x, 2)[0]                       # (B, G)
        return x
                
        
        