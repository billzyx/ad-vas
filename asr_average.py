import numpy as np
from ml_train import load_data_and_label

feature_dict, label_list = load_data_and_label()

asr_feature = feature_dict['ASR']
asr_feature = asr_feature[:, 0]

print(np.mean(asr_feature[label_list == 1]))
print(np.mean(asr_feature[label_list == 0]))
