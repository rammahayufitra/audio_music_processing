import os 
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset  

class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, max_audio_length):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.max_audio_length = max_audio_length

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        # Ensure the signal has a consistent length
        # signal = self._process_signal_length(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate: 
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal): 
        if signal.shape[0] > 1: 
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _process_signal_length(self, signal):
        # Truncate or zero-pad the signal to ensure consistent length
        if signal.size(1) > self.max_audio_length:
            signal = signal[:, :self.max_audio_length]
        elif signal.size(1) < self.max_audio_length:
            pad_amount = self.max_audio_length - signal.size(1)
            signal = torch.nn.functional.pad(signal, (0, pad_amount))
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index): 
        return self.annotations.iloc[index, 6]
    

if __name__ == "__main__": 
    ANNOTATION_FILE = "../data/URBANSOUND8K/UrbanSound8K.csv"
    AUDIO_DIR = "../data/URBANSOUND8K"
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 16000  # Adjust this value according to your maximum desired audio length

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024, 
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, MAX_AUDIO_LENGTH)
    print(f"There are {len(usd)} samples in the dataset.")

    signal, label = usd[0]

    print(signal, label)
