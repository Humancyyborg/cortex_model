import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import kurtosis, skew
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time

# 1. Generate fictitious EEG dataset
np.random.seed(42)

def generate_eeg_data(n_samples, n_points, condition):
    if condition == 'normal':
        data = np.random.normal(0, 100, (n_samples, n_points))
    elif condition == 'interictal':
        data = np.random.normal(0, 150, (n_samples, n_points))
        data += np.sin(np.linspace(0, 10*np.pi, n_points)) * 50
    elif condition == 'ictal':
        data = np.random.normal(0, 500, (n_samples, n_points))
        data += np.sin(np.linspace(0, 40*np.pi, n_points)) * 200
    return data

n_points = 4096
sampling_rate = 173.61

normal_data = generate_eeg_data(200, n_points, 'normal')
interictal_data = generate_eeg_data(200, n_points, 'interictal')
ictal_data = generate_eeg_data(100, n_points, 'ictal')

# 2. Implement feature extraction methods

def power_spectral_features(eeg_segment, sampling_rate):
    fft_vals = np.abs(fft(eeg_segment))
    freq = np.fft.fftfreq(len(eeg_segment), 1/sampling_rate)
    
    psi = []
    for i in range(1, 16):
        f_min, f_max = 2*i, 2*(i+1)
        idx_range = np.where((freq >= f_min) & (freq < f_max))
        psi.append(np.sum(fft_vals[idx_range]))
    
    total_power = np.sum(psi)
    rir = [p/total_power for p in psi]
    
    return psi + rir

def petrosian_fd(eeg_segment):
    diff = np.diff(eeg_segment)
    N = len(eeg_segment)
    N_delta = np.sum(diff[:-1] * diff[1:] < 0)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))

def higuchi_fd(eeg_segment, k_max=5):
    N = len(eeg_segment)
    L = np.zeros((k_max,))
    x = np.arange(1, k_max + 1)
    
    for k in range(1, k_max + 1):
        Lk = np.zeros((k,))
        for m in range(k):
            Lmk = 0
            for i in range(1, int((N-m)/k)):
                Lmk += abs(eeg_segment[m+i*k] - eeg_segment[m+(i-1)*k])
            Lmk = (Lmk * (N - 1) / (((N - m) / k) * k)) / k
            Lk[m] = Lmk
        L[k-1] = np.mean(Lk)
    
    return np.polyfit(np.log(x), np.log(L), 1)[0]

def hjorth_params(eeg_segment):
    diff1 = np.diff(eeg_segment)
    diff2 = np.diff(diff1)
    
    activity = np.var(eeg_segment)
    mobility = np.sqrt(np.var(diff1) / activity)
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
    
    return [mobility, complexity]

def extract_features(eeg_segment, sampling_rate):
    features = []
    
    # Power Spectral Features
    features.extend(power_spectral_features(eeg_segment, sampling_rate))
    
    # Fractal Dimensions
    features.append(petrosian_fd(eeg_segment))
    features.append(higuchi_fd(eeg_segment))
    
    # Hjorth Parameters
    features.extend(hjorth_params(eeg_segment))
    
    # Statistical Features
    features.extend([np.mean(eeg_segment), np.std(eeg_segment),
                     np.mean(np.abs(eeg_segment)), np.std(np.abs(eeg_segment))])
    
    return np.array(features)

# 3. Preprocess data and extract features

def preprocess_and_extract(data, sampling_rate):
    features = np.array([extract_features(segment, sampling_rate) for segment in data])
    
    # Standardize features
    scaler = StandardScaler()
    return scaler.fit_transform(features)

normal_features = preprocess_and_extract(normal_data, sampling_rate)
interictal_features = preprocess_and_extract(interictal_data, sampling_rate)
ictal_features = preprocess_and_extract(ictal_data, sampling_rate)

# PNN Implementation

class PNN:
    def __init__(self, spread=0.1):
        self.spread = spread
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            distances = np.sum((self.X_train - X[i])**2, axis=1)
            rbf_outputs = np.exp(-(distances) / (2 * self.spread**2))
            
            class_scores = np.zeros(len(self.classes))
            for j, c in enumerate(self.classes):
                class_scores[j] = np.sum(rbf_outputs[self.y_train == c])
            
            y_pred[i] = self.classes[np.argmax(class_scores)]
        
        return y_pred

# 4. Run experiments

def run_experiment(X, y):
    kf = KFold(n_splits=10)
    pnn = PNN(spread=0.1)
    
    predictions = []
    true_labels = []
    total_time = 0
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        pnn.fit(X_train, y_train)
        
        start_time = time.time()
        pred = pnn.predict(X_test)
        total_time += time.time() - start_time
        
        predictions.extend(pred)
        true_labels.extend(y_test)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    avg_time = total_time / len(X)
    
    return accuracy, precision, recall, f1, avg_time

# Prepare datasets for experiments
normal_interictal_data = np.vstack((normal_features, interictal_features))
normal_interictal_labels = np.array([0]*200 + [1]*200)

normal_ictal_data = np.vstack((normal_features, ictal_features))
normal_ictal_labels = np.array([0]*200 + [1]*100)

interictal_ictal_data = np.vstack((interictal_features, ictal_features))
interictal_ictal_labels = np.array([0]*200 + [1]*100)

zone_localization_data = interictal_features
zone_localization_labels = np.array([0]*100 + [1]*100)

experiments = [
    ("Normal vs Interictal", normal_interictal_data, normal_interictal_labels),
    ("Normal vs Ictal", normal_ictal_data, normal_ictal_labels),
    ("Interictal vs Ictal", interictal_ictal_data, interictal_ictal_labels),
    ("Epileptogenic zone vs Opposite hemisphere", zone_localization_data, zone_localization_labels)
]

# 5. Compare results
print("Experiment Results:")
print("------------------")
for name, X, y in experiments:
    accuracy, precision, recall, f1, avg_time = run_experiment(X, y)
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Avg. Classification Time: {avg_time:.3f} seconds")
    print()

print("Original Paper Results:")
print("-----------------------")
print("Normal vs Interictal: Accuracy = 99.5%, Time = 0.01s")
print("Normal vs Ictal: Accuracy = 98.3%, Time = 0.01s")
print("Interictal vs Ictal: Accuracy = 96.7%, Time = 0.01s")
print("Epileptogenic zone vs Opposite hemisphere: Accuracy = 77.5%, Time = 0.01s")
