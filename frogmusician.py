import torch
from torch import nn
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import librosa
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

class MusicAnalyzer:
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.aif', '.aiff']
    
    # Add task definitions
    TASKS = {
        'GS': ['B minor', 'D major', 'G major', 'D minor', 'A minor'],
        'MTGInstrument': ['synthesizer', 'guitar', 'electricguitar', 'drums', 'bass'],
        'MTGGenre': ['electronic', 'trance', 'pop', 'dance', 'progressive'],
        'MTGTop50': ['electronic', 'guitar', 'downtempo', 'chillout', 'bass'],
        'MTGMood': ['deep', 'summer', 'love', 'energetic', 'dream'],
        'NSynthI': ['guitar', 'brass', 'keyboard', 'flute', 'reed'],
        'NSynthP': ['26', '25', '36', '37', '27'],
        'VocalSetS': ['m10', 'm4', 'm7', 'm1', 'f3'],
        'VocalSetT': ['trillo', 'trill', 'lip_trill', 'inhaled', 'spoken'],
        'EMO': ['arousal', 'valence']
    }

    def __init__(self, db_path="music_features.db"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.resample_rate = self.processor.sampling_rate
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Create a simple database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS songs (
                        id INTEGER PRIMARY KEY,
                        title TEXT,
                        file_path TEXT UNIQUE,
                        format TEXT,
                        features BLOB,
                        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise

    def process_audio(self, file_path):
        try:
            # Load and resample audio
            audio, sr = librosa.load(file_path, sr=self.resample_rate)
            input_audio = torch.FloatTensor(audio).to(self.device)
            
            # Process through model
            inputs = self.processor(input_audio, sampling_rate=self.resample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Get all layer representations [25 layers, Time steps, 1024 features]
            all_layers = torch.stack(outputs.hidden_states).squeeze()
            
            # Store both time-reduced and full features
            time_reduced = all_layers.mean(-2).cpu().numpy()  # [25, 1024]
            full_features = all_layers.cpu().numpy()  # Full representation
            
            return {
                'time_reduced': time_reduced,
                'full_features': full_features
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def process_folder(self, folder_path):
        """Process all supported audio files in a folder."""
        # Ensure database exists
        self.setup_database()
        
        folder = Path(folder_path)
        audio_files = []
        for format in self.SUPPORTED_FORMATS:
            audio_files.extend(folder.glob(f"*{format}"))
        
        if not audio_files:
            print(f"No supported audio files found in {folder_path}")
            return
            
        with sqlite3.connect(self.db_path) as conn:
            for audio_file in tqdm(audio_files, desc="Processing audio files"):
                try:
                    features = self.process_audio(audio_file)
                    if features is not None:
                        # Convert features to bytes for storage
                        feature_bytes = np.save(BytesIO(), features)
                        conn.execute(
                            "INSERT OR REPLACE INTO songs (title, file_path, format, features) VALUES (?, ?, ?, ?)",
                            (audio_file.stem, 
                             str(audio_file), 
                             audio_file.suffix.lower(),
                             feature_bytes.getvalue())
                        )
                        conn.commit()
                        print(f"Processed: {audio_file.name}")
                except Exception as e:
                    print(f"Error with {audio_file}: {str(e)}")
                    continue

    def interpret_features(self, features):
        """Convert MERT features to task-specific predictions"""
        # Simplified example - in reality you'd need proper model heads for each task
        predictions = {}
        for task, classes in self.TASKS.items():
            # Simulate predictions using feature patterns
            task_features = features['time_reduced'].mean(axis=0)[:len(classes)]
            probs = np.exp(task_features) / np.sum(np.exp(task_features))
            predictions[task] = {cls: f"{prob:.2f}%" for cls, prob in zip(classes, probs * 100)}
        return predictions

    @staticmethod
    def view_features():
        st.set_page_config(page_title="MERT Music Analysis", layout="wide")
        st.title("MERT Feature Analysis")

        try:
            conn = sqlite3.connect("music_features.db")
            df = pd.read_sql("SELECT * FROM songs", conn)
            
            for idx, row in df.iterrows():
                with st.expander(f"Analysis for: {row['title']}"):
                    features = np.load(BytesIO(row['features']))
                    
                    # Get predictions
                    predictions = MusicAnalyzer.interpret_features(features)
                    
                    # Display predictions as a formatted table
                    st.subheader("Music Analysis Results")
                    results_df = pd.DataFrame({
                        'Task': [],
                        'Top 1': [], 'Top 2': [], 'Top 3': [], 'Top 4': [], 'Top 5': []
                    })
                    
                    for task, preds in predictions.items():
                        sorted_preds = sorted(preds.items(), key=lambda x: float(x[1].strip('%')), reverse=True)
                        row = {f'Top {i+1}': f"{cls} {prob}" for i, (cls, prob) in enumerate(sorted_preds[:5])}
                        row['Task'] = task
                        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                    
                    st.dataframe(results_df)
                    
                    # Original feature visualizations can follow...
                    # ...existing visualization code...
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyzer = MusicAnalyzer()
        analyzer.process_folder(sys.argv[1])
    else:
        MusicAnalyzer.view_features()