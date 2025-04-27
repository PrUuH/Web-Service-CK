import os
import sys
import wfdb
import heartpy as hp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import scipy.signal as ssignal
from scipy.fft import fft, fftfreq

# Определение модели 1D CNN
class Simple1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(8 * 9, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Добавляем канал
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Функция для извлечения признаков
def extract_features(signal, fs):
    try:
        wd, m = hp.process(signal, fs)
    except hp.exceptions.BadSignalWarning as e:
        print("BadSignalWarning:", e)
        return None
    
    # Вычисление дополнительных частотных характеристик
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)
    
    # Определение границ для LF и HF
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    # Вычисление LF и HF
    lf_mask = (xf >= lf_band[0]) & (xf <= lf_band[1])
    hf_mask = (xf >= hf_band[0]) & (xf <= hf_band[1])
    
    lf = np.trapz(np.abs(yf[lf_mask])**2, xf[lf_mask])
    hf = np.trapz(np.abs(yf[hf_mask])**2, xf[hf_mask])
    lf_hf_ratio = lf / hf if hf != 0 else 0
    
    # Сбор всех признаков
    features = {
        'bpm': m['bpm'],
        'ibi': m['ibi'],
        'sdnn': m['sdnn'],
        'sdsd': m['sdsd'],
        'rmssd': m['rmssd'],
        'pnn20': m['pnn20'],
        'pnn50': m['pnn50'],
        'hr_mad': m['hr_mad'],
        'sd1': m['sd1'],
        'sd2': m['sd2'],
        's': m['s'],
        'sd1/sd2': m['sd1/sd2'],
        'breathingrate': m['breathingrate'],
        'lf': lf,
        'hf': hf,
        'lf_hf_ratio': lf_hf_ratio
    }
    
    return features

# Главная функция
# Главная функция
def main(hea_file_path):
    # Загрузка записи
    record = wfdb.rdrecord(hea_file_path[:-4])  # Удаляем расширение .hea
    
    # Получение сигнала и частоты дискретизации
    signal = record.p_signal.astype(np.float32).flatten()
    fs = record.fs
    
    # Обрезаем сигнал до 5 минут
    five_minute_samples = 60 * 5 * fs
    signal = signal[:five_minute_samples]
    
    # Извлечение признаков
    features = extract_features(signal, fs)
    if features is None:
        return "Не удалось извлечь признаки."
    
    # Преобразование признаков в массив numpy
    feature_names = [
        'bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad',
        'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate', 'lf', 'hf', 'lf_hf_ratio'
    ]
    feature_values = [features[name] for name in feature_names]
    X = np.array(feature_values).reshape(1, -1)
    
    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Преобразование данных в тензор
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Инициализация модели
    input_dim = X_scaled.shape[1]
    num_classes = 15 
    model = Simple1DCNN(input_dim, num_classes)
    
    # Загрузка предобученных весов
    model.load_state_dict(torch.load('simple_1dcnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Предсказание
    with torch.no_grad():
        output = model(X_tensor)
        _, predicted = torch.max(output.data, 1)
    
    # Вывод результата
    age_group = predicted.item() + 1
    return f"Предсказанная возрастная группа: {age_group}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_hea_file>")
    else:
        hea_file_path = sys.argv[1]
        if not os.path.isfile(hea_file_path):
            print(f"File not found: {hea_file_path}")
        else:
            main(hea_file_path)