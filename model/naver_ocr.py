"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from model.ocr_modules.transformation import TPS_SpatialTransformerNetwork
from model.ocr_modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from model.ocr_modules.sequence_modeling import BidirectionalLSTM
from model.ocr_modules.prediction import Attention


class BasicModel(nn.Module):
    def __init__(self, Transformation, FeatureExtraction, SequenceModeling, Prediction,
                 num_class, num_fiducial=20, imgH=100, imgW=32, batch_max_length=25,
                 use_rgb=False, input_channel=1, output_channel=512, hidden_size=256):
        super(BasicModel, self).__init__()
        self.stages = {'Trans': Transformation, 'Feat': FeatureExtraction,
                       'Seq': SequenceModeling, 'Pred': Prediction}
        self.use_rgb = use_rgb
        self.batch_max_length = batch_max_length

        """ Transformation """
        if Transformation == 'TPS':
            raise NotImplementedError("Transformation is not supported")
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)
        else:
            # print('No Transformation module specified')
            pass

        """ FeatureExtraction """
        if FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.SequenceModeling_output = hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)
        elif Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text=None, is_train=True, normalize=True, return_feat=False):
        if not self.use_rgb:
            R, G, B = input[:,0:1,:,:], input[:,1:2,:,:], input[:,2:3,:,:]
            input = (0.299*R + 0.587*G + 0.114*B)  # luma input
        if normalize:
            input = (input - 0.5) / 0.5
        # feat_dict = dict()

        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input, batch_C_prime = self.Transformation(input)
            # feat_dict["batch_C_prime"] = batch_C_prime

        """ Feature extraction stage """
        visual_feature, feat_dict = self.FeatureExtraction(input)
        # feat_dict["visual_mid_feature"] = visual_mid_feature
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
        # feat_dict["contextual_feature"] = contextual_feature

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            if text is None:
                text = torch.LongTensor(input.size(0), self.batch_max_length + 1).fill_(0).to(input.device)
            else:
                text = text[:, :-1]  # align with Attention.forward
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.batch_max_length)

        if return_feat:
            return prediction, feat_dict
        else:
            return prediction
