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
import gc
import psutil
import math
import pandas as pd
from openpyxl.styles import PatternFill

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_torch_settings():
    """Configure PyTorch attention and memory settings"""
    if torch.cuda.is_available():
        # Check CUDA capabilities
        cuda_capability = torch.cuda.get_device_capability()
        supports_flash_attention = cuda_capability >= (8, 0)  # Ampere or newer GPUs
        
        if supports_flash_attention:
            print("Enabling Flash Attention")
            torch.backends.cuda.enable_flash_sdp(True)
        else:
            print("Flash Attention not supported on this GPU, using memory efficient attention")
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)
    else:
        print("Running on CPU - using default attention mechanism")

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
        setup_torch_settings()  # Configure attention mechanisms
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
            
    def analyze_audio(self, audio_path, chunk_duration=5, overlap=0.5):
        """Process audio in time-based chunks"""
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        
        # Calculate chunks based on duration
        stride_duration = chunk_duration - overlap
        num_chunks = math.ceil(duration / stride_duration)
        
        print(f"\nAudio stats:")
        print(f"Duration: {duration:.1f}s")
        print(f"Chunk size: {chunk_duration}s")
        print(f"Overlap: {overlap}s")
        print(f"Stride: {stride_duration}s")
        print(f"Number of chunks: {num_chunks}")
        
        # Convert time to samples for processing
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        stride_samples = chunk_samples - overlap_samples
        
        all_predictions = []
        for i in tqdm(range(num_chunks)):
            # Monitor VRAM
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024**2
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                print(f"\rVRAM usage: {vram_used:.0f}MB / {vram_total:.0f}MB", end="")
                
                # Clear CUDA cache if VRAM usage is high (>80%)
                if vram_used / vram_total > 0.99:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Extract chunk with overlap
            start = i * stride_samples
            end = min(start + chunk_samples, waveform.shape[1])
            chunk = waveform[:, start:end]
            
            # Process chunk
            try:
                predictions = self._process_single(chunk.reshape(-1))
                all_predictions.append(predictions)
            except RuntimeError as e:
                print(f"\nError processing chunk {i}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        # Average predictions
        return self._average_predictions(all_predictions)

    def _average_predictions(self, all_predictions):
        """Average predictions across chunks"""
        final_predictions = {}
        for task in self.TASKS:
            final_predictions[task] = {}
            for chunk_pred in all_predictions:
                for class_name, prob in chunk_pred[task].items():
                    prob_value = float(prob.strip('%'))
                    if class_name not in final_predictions[task]:
                        final_predictions[task][class_name] = prob_value
                    else:
                        final_predictions[task][class_name] += prob_value
                        
            # Average probabilities
            for class_name in final_predictions[task]:
                avg_prob = final_predictions[task][class_name] / len(all_predictions)
                final_predictions[task][class_name] = f"{avg_prob:.2f}%"
        
        return final_predictions

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

    def save_to_excel(self, predictions, song_name, output_dir='analysis_output'):
        """Save predictions to Excel file with multiple sheets"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'analysis_{song_name}_{timestamp}.xlsx'
        filepath = os.path.join(output_dir, filename)
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for task, results in predictions.items():
                # Convert task results to DataFrame
                df = pd.DataFrame({
                    'Class': list(results.keys()),
                    'Probability': [float(v.strip('%')) for v in results.values()]
                })
                
                # Sort by probability
                df = df.sort_values('Probability', ascending=False)
                
                # Write to Excel
                df.to_excel(writer, sheet_name=task, index=False)
                
                # Get worksheet for formatting
                worksheet = writer.sheets[task]
                
                # Auto-adjust column width
                for column in worksheet.columns:
                    max_length = 0
                    column = list(column)
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        print(f"Excel analysis saved to: {filepath}")
        return filepath

def process_songs_directory(songs_dir="songs", output_dir="analysis_output"):
    analyzer = MusicAnalyzer()
    audio_files = []
    
    for format in analyzer.SUPPORTED_FORMATS:
        audio_files.extend(Path(songs_dir).glob(f"*{format}"))
    
    if not audio_files:
        print(f"No supported audio files found in {songs_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in tqdm(audio_files, desc="Processing songs"):
        try:
            print(f"\nAnalyzing: {audio_file.name}")
            results = analyzer.analyze_audio(str(audio_file))
            analyzer.save_to_excel(results, audio_file.stem, output_dir)
            analyzer.visualize_results(results, output_dir, audio_file.name, output_dir)
        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
            continue

if __name__ == "__main__":
    process_songs_directory()