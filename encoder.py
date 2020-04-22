class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.resnet_encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.resnet_encoder(x)
        return x