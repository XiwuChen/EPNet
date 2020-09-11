# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import img_transform as T


def build_transforms(training):
    pixel_aug_prob = 0.2
    if training:
        transform = T.Compose(
            [
                T.RandomBrightness(pixel_aug_prob),
                T.RandomContrast(pixel_aug_prob),
                T.RandomHue(pixel_aug_prob),
                T.RandomSaturation(pixel_aug_prob),
                T.RandomGamma(pixel_aug_prob),
            ]
        )
    else:
        transform = T.Compose([])
    return transform
