import warnings
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor
import importlib
import json
import os
import re
from tqdm import tqdm
import time
import sqlite3
import numpy as np
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
from Prediction_Head.MTGGenre_head import MLPProberBase 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ERT_BEST_LAYER_IDX = {
    'EMO': 5,
    'GS': 8,
    'GTZAN': 7,
    'MTGGenre': 7,
    'MTGInstrument': 'all',
    'MTGMood': 6,
    'MTGTop50': 6,
    'MTT': 'all',
    'NSynthI': 6,
    'NSynthP': 1,
    'VocalSetS': 2,
    'VocalSetT': 9,
} 

MERT_BEST_LAYER_IDX = {
    'EMO': 5,
    'GS': 8,
    'GTZAN': 7,
    'MTGGenre': 7,
    'MTGInstrument': 'all',
    'MTGMood': 6,
    'MTGTop50': 6,
    'MTT': 'all',
    'NSynthI': 6,
    'NSynthP': 1,
    'VocalSetS': 2,
    'VocalSetT': 9,
} 
CLASSIFIERS = {

}

ID2CLASS = {

}

TASKS = ['GS', 'MTGInstrument', 'MTGGenre', 'MTGTop50', 'MTGMood', 'NSynthI', 'NSynthP', 'VocalSetS', 'VocalSetT','EMO',]
Regression_TASKS = ['EMO']
head_dir = './Prediction_Head/best-layer-MERT-v1-95M'
for task in TASKS:
    print('loading', task)
    with open(os.path.join(head_dir,f'{task}.id2class.json'), 'r') as f:
        ID2CLASS[task]=json.load(f)
    num_class = len(ID2CLASS[task].keys())
    CLASSIFIERS[task] = MLPProberBase(d=768, layer=MERT_BEST_LAYER_IDX[task], num_outputs=num_class)
    CLASSIFIERS[task].load_state_dict(torch.load(f'{head_dir}/{task}.ckpt')['state_dict'])
    CLASSIFIERS[task].to(device)

class MusicAnalyzer:
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.aif', '.aiff']
    
    def __init__(self, model_path="./MERT-v1-95M"):
        print("Initializing MusicAnalyzer...")
        self.setup_model(model_path)
        
    def setup_model(self, model_path):
        # Configure model
        modeling_MERT = importlib.import_module("MERT-v1-95M.modeling_MERT")
        config = modeling_MERT.MERTConfig.from_pretrained(model_path)
        config.conv_pos_batch_norm = False
        
        # Load model components
        self.model = modeling_MERT.MERTModel.from_pretrained(model_path, config=config)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        # Load classifiers
        self.load_classifiers()
        
    def load_classifiers(self):
        head_dir = './Prediction_Head/best-layer-MERT-v1-95M'
        self.TASKS = ['GS', 'MTGInstrument', 'MTGGenre', 'MTGTop50', 'MTGMood', 
                     'NSynthI', 'NSynthP', 'VocalSetS', 'VocalSetT', 'EMO']
        
        self.CLASSIFIERS = {}
        self.ID2CLASS = {}
        
        for task in self.TASKS:
            with open(os.path.join(head_dir, f'{task}.id2class.json'), 'r') as f:
                self.ID2CLASS[task] = json.load(f)
            num_class = len(self.ID2CLASS[task].keys())
            self.CLASSIFIERS[task] = MLPProberBase(d=768, layer=MERT_BEST_LAYER_IDX[task], 
                                                 num_outputs=num_class)
            self.CLASSIFIERS[task].load_state_dict(
                torch.load(f'{head_dir}/{task}.ckpt')['state_dict'])
            self.CLASSIFIERS[task].to(self.device)
            
    def analyze_audio(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        if not any(audio_path.lower().endswith(fmt) for fmt in self.SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported format. Supported: {self.SUPPORTED_FORMATS}")
            
        # Process audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != self.processor.sampling_rate:
            resampler = T.Resample(sample_rate, self.processor.sampling_rate)
            waveform = resampler(waveform)
            
        return self._process_single(waveform.reshape(-1))
        
    def _process_single(self, waveform):
        """Process a single chunk or short audio file"""
        model_inputs = self.processor(waveform, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**model_inputs, output_hidden_states=True)
        
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()[1:,:,:].unsqueeze(0)
        all_layer_hidden_states = all_layer_hidden_states.mean(dim=2)
        
        predictions = {}
        for task in self.TASKS:
            if MERT_BEST_LAYER_IDX[task] == 'all':
                logits = self.CLASSIFIERS[task](all_layer_hidden_states)
            else:
                logits = self.CLASSIFIERS[task](all_layer_hidden_states[:, MERT_BEST_LAYER_IDX[task]])
            
            sorted_idx = torch.argsort(logits, dim=-1, descending=True)[0]
            sorted_prob = nn.functional.softmax(logits[0], dim=-1)[sorted_idx]
            
            predictions[task] = {}
            top_n_show = min(5, len(self.ID2CLASS[task]))
            for idx in range(top_n_show):
                class_name = str(self.ID2CLASS[task][str(sorted_idx[idx].item())])
                class_name = re.sub(r'^\w+---', '', class_name)
                class_name = re.sub(r'^\w+\/\w+---', '', class_name)
                predictions[task][class_name] = f"{sorted_prob[idx].item():.2f}%"
        
        return predictions

    def visualize_results(self, predictions, output_dir='analysis_output'):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        for i, (task, results) in enumerate(predictions.items()):
            plt.subplot(3, 4, i+1)
            classes = list(results.keys())[:5]
            values = [float(v.strip('%')) for v in list(results.values())[:5]]
            plt.bar(classes, values)
            plt.title(task)
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'analysis_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        
        # Save plot
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Analysis saved to: {filepath}")
        return filepath

if __name__ == "__main__":
    analyzer = MusicAnalyzer()
    audio_path = "songs/bloody mary (I'll dance dance dance with my hands) - lady gaga [edit audio] [5pf9sGPZ5wU].wav"
    results = analyzer.analyze_audio(audio_path)
    
    # Save visualization
    analyzer.visualize_results(results)