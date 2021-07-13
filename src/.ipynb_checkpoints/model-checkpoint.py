import monai

def get_model(spatial_dims=3, in_channels=2, num_classes=1):
    return monai.networks.nets.se_resnext50_32x4d(spatial_dims=spatial_dims, in_channels=in_channels, num_classes=num_classes)