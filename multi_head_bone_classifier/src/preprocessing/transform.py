from torchvision import transforms


def get_train_transform(hparams):
    input_size = hparams.input_size[-1]
    augmentations = []
    augmentations.append(
        transforms.RandomResizedCrop(size=input_size, scale=(0.7, 1.0))
    )
    if hparams.aug_rotation:
        augmentations.append(transforms.RandomRotation(degrees=30))
    if hparams.aug_color_jitter:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
        )
    augmentations.append(transforms.Grayscale(num_output_channels=3))
    augmentations.append(transforms.ToTensor())
    train_transform = transforms.Compose(augmentations)
    return train_transform


def get_val_transform(hparams):
    input_size = hparams.input_size[-1]
    val_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    return val_transform


def get_test_transform(hparams):
    input_size = hparams.input_size[-1]
    test_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    return test_transform
