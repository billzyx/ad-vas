import os
from tqdm import tqdm
from recognize import ASR
import numpy as np


def process_path(path):
    asr = ASR(model_weight='../wav2vec/pre_train_weights/wav2vec_vox_960h_pl.pt', target_dict='../wav2vec/pre_train_weights/dict.ltr.txt')
    for name in tqdm(sorted(os.listdir(path))):
        if name.endswith('.wav'):
            file_path = os.path.join(path, name)
            text, audio_embedding = asr.predict_file(file_path)
            new_file_path = file_path.replace('vas-data', 'vas-data-asr').replace('.wav', '.txt')
            if not os.path.isdir(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))
            with open(new_file_path, 'w') as f:
                f.write(text)

            new_file_path = file_path.replace('vas-data', 'vas-data-asr').replace('.wav', '.npy')
            if not os.path.isdir(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))
            np.save(new_file_path, audio_embedding)


if __name__ == '__main__':
    full_wave_enhanced_audio_path = 'vas-data'
    process_path(full_wave_enhanced_audio_path)


