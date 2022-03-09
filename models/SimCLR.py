import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):

    def __init__(self, encoder_name, out_dim):
        super(SimCLR, self).__init__()

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        self.encoder = self.resnet_dict[encoder_name]
        in_features = self.encoder.fc.in_features
        self.proj_head = nn.Sequential(nn.Linear(in_features, in_features), # add Proj_Head
                                       nn.ReLU(),
                                       self.encoder.fc)

        self.encoder.fc = nn.Identity() # Remove classifier

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.proj_head(h_i)
        z_j = self.proj_head(h_j)

        return h_i, h_j, z_i, z_j


if __name__ == "__main__":
    a = torch.randn(2, 3, 256, 256).cuda()
    b = SimCLR('resnet50', out_dim=1024).cuda()
    print(b(a, a)[2].size())
