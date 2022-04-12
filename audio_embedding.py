import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import time
import copy
from transformers import AdamW
from torch import nn
import re
import itertools as it
import transformers
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.nn.utils.rnn import pad_sequence
import jiwer
import xlwt

feature_dir = 'feature'
if not os.path.isdir(feature_dir):
    os.makedirs(feature_dir)


def load_file(file_path, level_list=(), punctuation_list=()):
    text_file = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        sentence = lines[0]
        original_sentence, sentence = preprocess_text(sentence, level_list, punctuation_list)
        text_file['asr_text'] = sentence.lower()
        text_file['asr_original_text'] = original_sentence.lower()
    return text_file


def preprocess_text(sentence, level_list=(5, 10, 15, 30), punctuation_list=(',', '.', '..', '...')):
    sentence = sentence.strip().replace(" ", "").replace('<s>', '-')
    sentence = sentence.replace("--------------",
                                "-------------|")
    original_sentence = sentence
    new_sentence = []
    for g in it.groupby(list(sentence)):
        g_list = list(g[1])
        if g[0] != '|' and g[0] != '-':
            new_sentence.append(g[0])
        else:
            new_sentence.extend(g_list)
    sentence = ''.join(new_sentence)
    sentence = re.sub('\\b(-)+\\b', '', sentence)
    sentence = re.sub('\\b(-)+\'', '\'', sentence)
    sentence = sentence.replace("-", "|")
    while sentence.startswith('|'):
        sentence = sentence[1:]
    # sentence = re.sub('\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|'
    #                   '\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|+', '.... ', sentence)
    # sentence = re.sub('\|{30,}', '... ', sentence)
    assert len(level_list) == len(punctuation_list), 'level_list and punctuation_list must have the same length.'
    for level, punctuation in zip(reversed(level_list), reversed(punctuation_list)):
        sentence = re.sub('\|{' + str(level) + ',}', punctuation + ' ', sentence)
    sentence = re.sub('\|+', ' ', sentence)
    return original_sentence, sentence


def get_file_text(file_path):
    text_file = ''
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.strip().replace('	', ' ')
            if line.startswith('*'):
                text = text.split(':', maxsplit=1)[1] + ' '
                temp_idx = idx
                while not '' in lines[temp_idx]:
                    temp_idx += 1
                    text += lines[temp_idx].strip() + ' '
                text = text.split('')[0]
                text_file += text

    # print(text_file)
    text_file = text_file.replace('_', '')
    text_file = re.sub(r'\[[^\]]+\]', '', text_file)
    # text_file = re.sub('[^0-9a-zA-Z,. \'?]+', '', text_file)
    text_file = re.sub('[^0-9a-zA-Z \']+', '', text_file)
    # text_file = text_file.replace('...', '').replace('..', '')
    text_file = text_file.lower()
    # print(text_file)
    return text_file


def run_asr_text():
    data_path = 'vas-data-asr'
    text_dict = dict()
    for text_file_name in sorted(os.listdir(data_path)):
        if text_file_name.endswith('.txt'):
            text_file_path = os.path.join(data_path, text_file_name)
            text = load_file(text_file_path)
            text_dict[text_file_name.split('.')[0]] = text['text']
            print(text_file_path)
            print(text['text'])

    model_name = 'bert-base-uncased'
    configuration = transformers.BertConfig.from_pretrained(model_name)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    text_model = transformers.BertModel.from_pretrained(model_name, return_dict=True)

    embedding_list = []

    for key, text in text_dict.items():
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        text_outputs = text_model(**inputs)
        sequence_output = text_outputs[0]
        sequence_output = sequence_output.detach().numpy()
        avg_output = np.mean(sequence_output[0], axis=0)
        embedding_list.append(avg_output)

    embedding_list = np.array(embedding_list)
    k = 2
    kmeans = KMeans(n_clusters=k).fit(embedding_list)
    result = kmeans.labels_
    for i in range(len(text_dict)):
        print(list(text_dict.keys())[i], result[i])
    x_embedded = TSNE(n_components=2).fit_transform(embedding_list)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=result)
    plt.show()


def run_asr_audio_embedding():
    data_path = 'vas-data-asr'
    audio_dict = dict()
    for audio_file_name in sorted(os.listdir(data_path)):
        if audio_file_name.endswith('.npy'):
            audio_file_path = os.path.join(data_path, audio_file_name)
            audio = np.load(audio_file_path)
            audio_dict[audio_file_name.split('.')[0]] = audio

    audio_list = [torch.tensor(audio) for audio in audio_dict.values()]
    audio_list = pad_sequence(audio_list, batch_first=True).numpy()
    embedding_list = np.array(audio_list)
    embedding_list_shape = np.shape(embedding_list)
    embedding_list = np.reshape(
        embedding_list, [embedding_list_shape[0], embedding_list_shape[1] * embedding_list_shape[2]])
    k = 2
    kmeans = KMeans(n_clusters=k).fit(embedding_list)
    result = kmeans.labels_
    for i in range(len(audio_dict)):
        print(list(audio_dict.keys())[i], result[i])
    x_embedded = TSNE(n_components=2).fit_transform(embedding_list)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=result)
    plt.show()


def run_asr_text_wer():
    asr_data_path = 'vas-data-asr'
    transcript_data_path = 'vas-data'
    text_dict = dict()
    for text_file_name in sorted(os.listdir(asr_data_path)):
        if text_file_name.endswith('.txt'):
            text_file_path = os.path.join(asr_data_path, text_file_name)
            text = load_file(text_file_path)
            text_dict[text_file_name.split('.')[0]] = text
            print(text_file_path)
            print(text['asr_text'])

    for text_file_name in sorted(os.listdir(transcript_data_path)):
        if text_file_name.endswith('.cha'):
            text_file_path = os.path.join(transcript_data_path, text_file_name)
            text = get_file_text(text_file_path)
            text_dict[text_file_name.split('.')[0]]['transcript_text'] = text
            print(text_file_path)
            print(text)

    asr_label_dict = dict()
    for file_key, file_value in text_dict.items():
        asr_text = file_value['asr_text']
        transcript_text = file_value['transcript_text']
        wer = jiwer.wer(transcript_text, asr_text)
        measures = jiwer.compute_measures(transcript_text, asr_text)
        truth, hypothesis = jiwer.measures._preprocess(
            transcript_text, asr_text, jiwer.measures._default_transform, jiwer.measures._default_transform
        )
        normalized_wer = float(measures['substitutions'] + measures['deletions'] + measures['insertions']) \
                         / max(len(truth), len(hypothesis))
        mer = jiwer.mer(transcript_text, asr_text)
        print(file_key, wer, normalized_wer, mer)
        asr_label_dict[file_key] = [wer, normalized_wer, mer]
    write_xls(asr_label_dict)


def write_xls(label_dict):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet 1')

    label_name_list = ['WER', 'Normalized WER',
                       'MER', ]

    for idx, label_name in enumerate(label_name_list):
        ws.write(0, idx + 1, label_name)

    for row_idx, (key, label_list) in enumerate(label_dict.items()):
        ws.write(row_idx + 1, 0, key)
        for col_idx, label in enumerate(label_list):
            ws.write(row_idx + 1, col_idx + 1, label)
    wb.save(os.path.join(feature_dir, 'asr_view.xls'))


if __name__ == '__main__':
    # run_asr_text()
    # run_asr_audio_embedding()
    run_asr_text_wer()
