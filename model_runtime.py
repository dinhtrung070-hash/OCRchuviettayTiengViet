"""Construct the same VietOCR weights without VGG's unused ImageNet classifier.

Layer names, pooling, transformer and state-dict format match VietOCR 0.3.13.
Only feature layers are instantiated, reducing peak RAM on CPU machines.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.vgg import cfgs, make_layers
from vietocr.model.seqmodel.transformer import LanguageTransformer
from vietocr.model.vocab import Vocab
from vietocr.tool.predictor import Predictor as OriginalPredictor


class FeatureBody(nn.Module):
    def __init__(self, ss, ks, hidden, dropout=.5, **unused):
        super().__init__()
        self.features = make_layers(cfgs["E"], batch_norm=True)
        index = 0
        for i, module in enumerate(self.features):
            if isinstance(module, nn.MaxPool2d):
                self.features[i] = nn.AvgPool2d(kernel_size=ks[index], stride=ss[index])
                index += 1
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, image):
        features = self.last_conv_1x1(self.dropout(self.features(image)))
        return features.transpose(-1, -2).flatten(2).permute(-1, 0, 1)


class Backbone(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.model = FeatureBody(**config)

    def forward(self, image):
        return self.model(image)


class FormVietOCR(nn.Module):
    def __init__(self, size: int, config: dict):
        super().__init__()
        self.cnn = Backbone(**config["cnn"])
        self.seq_modeling = "transformer"
        self.transformer = LanguageTransformer(size, **config["transformer"])

    def forward(self, image, target, padding_mask):
        return self.transformer(self.cnn(image), target, tgt_key_padding_mask=padding_mask)


def build_model(config):
    if config["backbone"] != "vgg19_bn" or config["seq_modeling"] != "transformer" or config["cnn"].get("pretrained"):
        from vietocr.tool.translate import build_model as original_build
        return original_build(config)
    vocab = Vocab(config["vocab"])
    return FormVietOCR(len(vocab), config).to(config["device"]), vocab


class Predictor(OriginalPredictor):
    def __init__(self, config):
        self.model, self.vocab = build_model(config)
        source = config["weights"]
        if source.startswith("http"):
            from vietocr.tool.utils import download_weights
            source = download_weights(source)
        self.model.load_state_dict(torch.load(source, map_location=config["device"], weights_only=True))
        self.config = config
        self.device = config["device"]
