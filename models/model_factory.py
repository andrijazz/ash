import torchvision.transforms as transforms

from models.densenet import DenseNet3
from models.mobilenetv2 import MobileNetV2
from models.resnet_cifar import ResNet34
from models.resnet_imagenet import ResNet50, ResNet101


def build_model(model_name, num_classes):
    if model_name == 'resnet34':
        model = ResNet34(num_classes)
        transform = transforms.Compose([
            transforms.CenterCrop(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return model, transform
    if model_name == 'resnet50':
        model = ResNet50()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'resnet101':
        model = ResNet101()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'densenet100':
        model = DenseNet3(100, int(num_classes))
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
        ])
        return model, transform
    exit('{} model is not supported'.format(model_name))
