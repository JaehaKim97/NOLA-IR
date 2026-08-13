import torch
import torch.nn as nn


def LabelConverter(character, batch_max_length, type="ctc"):
    if type == "ctc":
        return CTCLabelConverter(character)
    elif type == "attn":
        return AttnLabelConverter(character, batch_max_length)
    else:
        raise NotImplementedError(f"Label converter type {type} is not implemented")


class CTCLabelConverter(nn.Module):
    """ Convert between text-label and text-index """
    def __init__(self, character):
        super().__init__()
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)
        self.register_buffer('device_indicator', torch.zeros(1))  # a dummy parameter to track device

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)

        device = self.device_indicator.device
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class AttnLabelConverter(nn.Module):
    """ Convert between text-label and text-index """

    def __init__(self, character, batch_max_length):
        super().__init__()
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        self.batch_max_length = batch_max_length

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i
        self.register_buffer('device_indicator', torch.zeros(1))  # a dummy parameter to track device

    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length = self.batch_max_length
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        device = self.device_indicator.device
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


# def calculate_char_accuracy(pred: str, gt: str) -> float:
#     max_len = max(len(pred), len(gt))
#     if max_len == 0:
#         return 1.0
#     matched = sum(p == g for p, g in zip(pred, gt))
#     return matched / max_len


def calculate_char_accuracy(pred: list[str], gt: list[str]) -> list[float]:
    accuracies = []
    for p, g in zip(pred, gt):
        max_len = max(len(p), len(g))
        if max_len == 0:
            accuracies.append(1.0)
        else:
            matched = sum(pc == gc for pc, gc in zip(p, g))
            accuracies.append(matched / max_len)
    return accuracies
