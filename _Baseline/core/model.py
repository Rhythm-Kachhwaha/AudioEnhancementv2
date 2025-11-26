import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import STFT, InverseSTFT
import nemo.collections.asr as nemo_asr
import tempfile
import soundfile as sf
import numpy as np
import os
import shutil

class VoiceFilter(nn.Module):
    def __init__(self, embedding_dim=192, hidden_size=256, num_layers=2):
        super(VoiceFilter, self).__init__()
        
        # CNN for spectrogram processing
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # LSTM for temporal processing
        # After 3 MaxPool2d(2) operations: 257 -> 128 -> 64 -> 32
        # So CNN output will be [batch, 64_channels, 32_freq, time]
        # Flattened: 64 * 32 = 2048 features + 192 embedding = 2240 total
        self.lstm = nn.LSTM(input_size=2048 + embedding_dim,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True)
        
        # Fully connected layers for mask prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spectrogram, speaker_embedding):
        # Ensure speaker embedding is properly shaped [batch_size, embedding_dim]
        if len(speaker_embedding.shape) > 2:
            speaker_embedding = speaker_embedding.view(speaker_embedding.size(0), -1)
        
        # Process spectrogram with CNN
        x = spectrogram.unsqueeze(1)  # Add channel dim
        x = self.cnn(x)
        
        # Reshape for LSTM (batch, time, features)
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 2, 1)  # (batch, time, freq, channels)
        x = x.reshape(batch, time, freq * channels)
        
        # Expand speaker embedding to match time dimension
        speaker_embedding = speaker_embedding.unsqueeze(1).expand(-1, time, -1)
        
        # Concatenate spectrogram features with speaker embedding
        x = torch.cat([x, speaker_embedding], dim=-1)
        
        # Process with LSTM
        x, _ = self.lstm(x)
        
        # Predict mask
        mask = self.fc(x)
        
        # Reshape mask to match original spectrogram dimensions
        mask = mask.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, freq, time)
        mask = F.interpolate(mask, size=spectrogram.shape[-2:], mode='bilinear')
        
        return mask.squeeze(1)

class SpeakerAwareEnhancer(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400):
        super(SpeakerAwareEnhancer, self).__init__()
        self.sample_rate = sample_rate
        
        # Load pretrained ECAPA-TDNN speaker encoder
        self.speaker_encoder = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='ecapa_tdnn'
        )
        
        # Freeze speaker encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        self.speaker_encoder.eval()
            
        # VoiceFilter model
        self.voice_filter = VoiceFilter()
        
        # STFT transforms
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.istft = InverseSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        
        # Create a temporary directory for this session
        self.temp_dir = tempfile.mkdtemp()
        
    def __del__(self):
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
    
    def _process_single_audio(self, audio_np, sample_idx):
        """Process a single audio sample and return embedding"""
        try:
            # Ensure we have a proper numpy array
            if torch.is_tensor(audio_np):
                if audio_np.is_cuda:
                    audio_np = audio_np.detach().cpu().numpy()
                else:
                    audio_np = audio_np.detach().numpy()
            
            # Ensure audio is 1D and properly formatted
            if len(audio_np.shape) > 1:
                audio_np = audio_np.squeeze()
            
            # Convert to proper numpy array if needed
            audio_np = np.asarray(audio_np, dtype=np.float32)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / np.max(np.abs(audio_np))
            else:
                return np.zeros(192, dtype=np.float32)
            
            # Create temporary file path
            temp_file = os.path.join(self.temp_dir, f"temp_audio_{sample_idx}.wav")
            
            # Write audio file
            sf.write(temp_file, 
                    audio_np.astype(np.float32), 
                    self.sample_rate,
                    subtype='PCM_16')
            
            # Get embedding
            embedding = self.speaker_encoder.get_embedding(temp_file)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # Ensure embedding is numpy array and flatten any extra dimensions
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten()
            else:
                embedding = np.array(embedding).flatten()
            
            # Ensure it's the right size (192 for ECAPA-TDNN)
            if embedding.shape[0] != 192:
                if embedding.shape[0] < 192:
                    embedding = np.pad(embedding, (0, 192 - embedding.shape[0]))
                else:
                    embedding = embedding[:192]
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Error processing audio sample {sample_idx}: {e}")
            print(f"Audio type: {type(audio_np)}, Audio shape: {getattr(audio_np, 'shape', 'No shape')}")
            if torch.is_tensor(audio_np):
                print(f"Tensor device: {audio_np.device}")
            return np.zeros(192, dtype=np.float32)
        
    def get_speaker_embeddings(self, reference_audio_batch):
        """Extract speaker embeddings for a batch of reference audio"""
        batch_size = reference_audio_batch.size(0)
        embeddings = []
        original_device = reference_audio_batch.device
        
        with torch.no_grad():
            # Move to CPU and ensure it's properly detached from any gradients
            if reference_audio_batch.is_cuda:
                ref_audio_cpu = reference_audio_batch.detach().cpu().clone().numpy()
            else:
                ref_audio_cpu = reference_audio_batch.detach().clone().numpy()
            
            for i in range(batch_size):
                # Make sure we're working with a clean numpy array
                audio_sample = ref_audio_cpu[i].copy()
                embedding = self._process_single_audio(audio_sample, i)
                embeddings.append(embedding)
        
        # Convert to tensor and move back to original device
        embeddings = np.stack(embeddings)
        embeddings = torch.from_numpy(embeddings).to(original_device)
        
        return embeddings
        
    def forward(self, reference_audio, noisy_audio):
        # Extract speaker embedding from reference audio
        speaker_embeddings = self.get_speaker_embeddings(reference_audio)
        
        # Compute STFT of noisy audio
        noisy_spec = self.stft(noisy_audio)
        
        # Handle complex STFT output properly
        if torch.is_complex(noisy_spec):
            noisy_mag = torch.abs(noisy_spec)
        else:
            noisy_mag = torch.sqrt(noisy_spec.pow(2).sum(-1))
        
        # Predict mask using VoiceFilter
        mask = self.voice_filter(noisy_mag, speaker_embeddings)
        
        # Apply mask to noisy spectrogram
        if torch.is_complex(noisy_spec):
            # For complex spectrograms, apply mask to magnitude and preserve phase
            phase = torch.angle(noisy_spec)
            masked_magnitude = noisy_mag * mask
            enhanced_spec = masked_magnitude * torch.exp(1j * phase)
        else:
            enhanced_spec = noisy_spec * mask.unsqueeze(-1)
        
        # Reconstruct enhanced audio
        enhanced_audio = self.istft(enhanced_spec)
        
        return enhanced_audio, mask
    
    def compute_loss(self, enhanced_audio, clean_audio):
        # Ensure same length
        min_len = min(enhanced_audio.size(-1), clean_audio.size(-1))
        enhanced_audio = enhanced_audio[..., :min_len]
        clean_audio = clean_audio[..., :min_len]
        
        # Compute STFT of clean audio
        clean_spec = self.stft(clean_audio)
        if torch.is_complex(clean_spec):
            clean_mag = torch.abs(clean_spec)
        else:
            clean_mag = torch.sqrt(clean_spec.pow(2).sum(-1))
        
        # Compute STFT of enhanced audio
        enhanced_spec = self.stft(enhanced_audio)
        if torch.is_complex(enhanced_spec):
            enhanced_mag = torch.abs(enhanced_spec)
        else:
            enhanced_mag = torch.sqrt(enhanced_spec.pow(2).sum(-1))
        
        # Magnitude spectrum loss
        mag_loss = F.mse_loss(enhanced_mag, clean_mag)
        
        # Waveform loss
        waveform_loss = F.mse_loss(enhanced_audio, clean_audio)
        
        # Combined loss
        total_loss = mag_loss + 0.1 * waveform_loss
        
        return total_loss



# above one should work with gpu




















# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils import STFT, InverseSTFT
# import nemo.collections.asr as nemo_asr
# import tempfile
# import soundfile as sf
# import numpy as np
# import os
# import shutil

# class VoiceFilter(nn.Module):
#     def __init__(self, embedding_dim=192, hidden_size=256, num_layers=2):
#         super(VoiceFilter, self).__init__()
        
#         # CNN for spectrogram processing
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
        
#         # LSTM for temporal processing
#         # After 3 MaxPool2d(2) operations: 257 -> 128 -> 64 -> 32
#         # So CNN output will be [batch, 64_channels, 32_freq, time]
#         # Flattened: 64 * 32 = 2048 features + 192 embedding = 2240 total
#         self.lstm = nn.LSTM(input_size=2048 + embedding_dim,  # Updated to match actual CNN output
#                            hidden_size=hidden_size,
#                            num_layers=num_layers,
#                            batch_first=True,
#                            bidirectional=True)
        
#         # Fully connected layers for mask prediction
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, spectrogram, speaker_embedding):
#         # Ensure speaker embedding is properly shaped [batch_size, embedding_dim]
#         if len(speaker_embedding.shape) > 2:
#             # Flatten any extra dimensions
#             speaker_embedding = speaker_embedding.view(speaker_embedding.size(0), -1)
        
#         # Debug print to check dimensions
#         print(f"Spectrogram shape: {spectrogram.shape}")
#         print(f"Speaker embedding shape: {speaker_embedding.shape}")
        
#         # Process spectrogram with CNN
#         x = spectrogram.unsqueeze(1)  # Add channel dim
#         x = self.cnn(x)
        
#         # Reshape for LSTM (batch, time, features)
#         batch, channels, freq, time = x.size()
#         x = x.permute(0, 3, 2, 1)  # (batch, time, freq, channels)
#         x = x.reshape(batch, time, freq * channels)
        
#         print(f"After CNN reshape: {x.shape}")
        
#         # Expand speaker embedding to match time dimension
#         # speaker_embedding should be [batch, embedding_dim]
#         speaker_embedding = speaker_embedding.unsqueeze(1).expand(-1, time, -1)
        
#         print(f"Speaker embedding after expand: {speaker_embedding.shape}")
        
#         # Concatenate spectrogram features with speaker embedding
#         x = torch.cat([x, speaker_embedding], dim=-1)
        
#         print(f"After concatenation: {x.shape}")
        
#         # Process with LSTM
#         x, _ = self.lstm(x)
        
#         # Predict mask
#         mask = self.fc(x)
        
#         # Reshape mask to match original spectrogram dimensions
#         mask = mask.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, freq, time)
#         mask = F.interpolate(mask, size=spectrogram.shape[-2:], mode='bilinear')
        
#         return mask.squeeze(1)

# class SpeakerAwareEnhancer(nn.Module):
#     def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400):
#         super(SpeakerAwareEnhancer, self).__init__()
#         self.sample_rate = sample_rate
        
#         # Load pretrained ECAPA-TDNN speaker encoder
#         self.speaker_encoder = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
#             model_name='ecapa_tdnn'
#         )
        
#         # Freeze speaker encoder
#         for param in self.speaker_encoder.parameters():
#             param.requires_grad = False
#         self.speaker_encoder.eval()  # Set to eval mode
            
#         # VoiceFilter model
#         self.voice_filter = VoiceFilter()
        
#         # STFT transforms
#         self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
#         self.istft = InverseSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        
#         # Create a temporary directory for this session
#         self.temp_dir = tempfile.mkdtemp()
        
#     def __del__(self):
#         # Clean up temporary directory
#         try:
#             shutil.rmtree(self.temp_dir, ignore_errors=True)
#         except:
#             pass
    
#     def _process_single_audio(self, audio_np, sample_idx):
#         """Process a single audio sample and return embedding"""
#         try:
#             # Ensure audio is 1D and properly formatted
#             if len(audio_np.shape) > 1:
#                 audio_np = audio_np.squeeze()
            
#             # Normalize audio to prevent clipping
#             if np.max(np.abs(audio_np)) > 0:
#                 audio_np = audio_np / np.max(np.abs(audio_np))
#             else:
#                 # Return zero embedding for silent audio
#                 return np.zeros(192, dtype=np.float32)
            
#             # Create temporary file path
#             temp_file = os.path.join(self.temp_dir, f"temp_audio_{sample_idx}.wav")
            
#             # Write audio file
#             sf.write(temp_file, 
#                     audio_np.astype(np.float32), 
#                     self.sample_rate,
#                     subtype='PCM_16')
            
#             # Get embedding
#             embedding = self.speaker_encoder.get_embedding(temp_file)
            
#             # Clean up temporary file immediately
#             os.remove(temp_file)
            
#             # Ensure embedding is numpy array and flatten any extra dimensions
#             if isinstance(embedding, np.ndarray):
#                 embedding = embedding.flatten()
#             else:
#                 embedding = np.array(embedding).flatten()
            
#             # Ensure it's the right size (192 for ECAPA-TDNN)
#             if embedding.shape[0] != 192:
#                 print(f"Warning: Unexpected embedding size {embedding.shape}, padding/truncating to 192")
#                 if embedding.shape[0] < 192:
#                     # Pad with zeros
#                     embedding = np.pad(embedding, (0, 192 - embedding.shape[0]))
#                 else:
#                     # Truncate
#                     embedding = embedding[:192]
            
#             return embedding.astype(np.float32)
            
#         except Exception as e:
#             print(f"Warning: Error processing audio sample {sample_idx}: {e}")
#             # Return zero embedding as fallback
#             return np.zeros(192, dtype=np.float32)
        
#     def get_speaker_embeddings(self, reference_audio_batch):
#         """Extract speaker embeddings for a batch of reference audio"""
#         batch_size = reference_audio_batch.size(0)
#         embeddings = []
        
#         # Move to CPU for processing
#         ref_audio_cpu = reference_audio_batch.cpu().numpy()
        
#         with torch.no_grad():
#             for i in range(batch_size):
#                 embedding = self._process_single_audio(ref_audio_cpu[i], i)
#                 embeddings.append(embedding)
        
#         # Convert to tensor and move to correct device
#         embeddings = np.stack(embeddings)
#         embeddings = torch.from_numpy(embeddings).to(reference_audio_batch.device)
        
#         print(f"Final embeddings shape: {embeddings.shape}")
        
#         return embeddings
        
#     def forward(self, reference_audio, noisy_audio):
#         # Extract speaker embedding from reference audio
#         speaker_embeddings = self.get_speaker_embeddings(reference_audio)
        
#         # Compute STFT of noisy audio
#         noisy_spec = self.stft(noisy_audio)
        
#         # Handle complex STFT output properly
#         if torch.is_complex(noisy_spec):
#             noisy_mag = torch.abs(noisy_spec)
#         else:
#             # If STFT returns real values, compute magnitude
#             noisy_mag = torch.sqrt(noisy_spec.pow(2).sum(-1))
        
#         print(f"Noisy magnitude shape: {noisy_mag.shape}")
        
#         # Predict mask using VoiceFilter
#         mask = self.voice_filter(noisy_mag, speaker_embeddings)
        
#         # Apply mask to noisy spectrogram
#         if torch.is_complex(noisy_spec):
#             # For complex spectrograms, apply mask to magnitude and preserve phase
#             phase = torch.angle(noisy_spec)
#             masked_magnitude = noisy_mag * mask
#             enhanced_spec = masked_magnitude * torch.exp(1j * phase)
#         else:
#             enhanced_spec = noisy_spec * mask.unsqueeze(-1)
        
#         # Reconstruct enhanced audio
#         enhanced_audio = self.istft(enhanced_spec)
        
#         return enhanced_audio, mask
    
#     def compute_loss(self, enhanced_audio, clean_audio):
#         # Ensure same length
#         min_len = min(enhanced_audio.size(-1), clean_audio.size(-1))
#         enhanced_audio = enhanced_audio[..., :min_len]
#         clean_audio = clean_audio[..., :min_len]
        
#         # Compute STFT of clean audio
#         clean_spec = self.stft(clean_audio)
#         if torch.is_complex(clean_spec):
#             clean_mag = torch.abs(clean_spec)
#         else:
#             clean_mag = torch.sqrt(clean_spec.pow(2).sum(-1))
        
#         # Compute STFT of enhanced audio
#         enhanced_spec = self.stft(enhanced_audio)
#         if torch.is_complex(enhanced_spec):
#             enhanced_mag = torch.abs(enhanced_spec)
#         else:
#             enhanced_mag = torch.sqrt(enhanced_spec.pow(2).sum(-1))
        
#         # Magnitude spectrum loss
#         mag_loss = F.mse_loss(enhanced_mag, clean_mag)
        
#         # Waveform loss
#         waveform_loss = F.mse_loss(enhanced_audio, clean_audio)
        
#         # Combined loss
#         total_loss = mag_loss + 0.1 * waveform_loss  # Weight the waveform loss
        
#         return total_loss

























# # below one is woking with cpu
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils import STFT, InverseSTFT
# import nemo.collections.asr as nemo_asr
# import tempfile
# import soundfile as sf
# import numpy as np
# import os
# import shutil

# class VoiceFilter(nn.Module):
#     def __init__(self, embedding_dim=192, hidden_size=256, num_layers=2):
#         super(VoiceFilter, self).__init__()
        
#         # CNN for spectrogram processing
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
        
#         # LSTM for temporal processing
#         # After 3 MaxPool2d(2) operations: 257 -> 128 -> 64 -> 32
#         # So CNN output will be [batch, 64_channels, 32_freq, time]
#         # Flattened: 64 * 32 = 2048 features + 192 embedding = 2240 total
#         self.lstm = nn.LSTM(input_size=2048 + embedding_dim,
#                            hidden_size=hidden_size,
#                            num_layers=num_layers,
#                            batch_first=True,
#                            bidirectional=True)
        
#         # Fully connected layers for mask prediction
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, spectrogram, speaker_embedding):
#         # Ensure speaker embedding is properly shaped [batch_size, embedding_dim]
#         if len(speaker_embedding.shape) > 2:
#             speaker_embedding = speaker_embedding.view(speaker_embedding.size(0), -1)
        
#         # Process spectrogram with CNN
#         x = spectrogram.unsqueeze(1)  # Add channel dim
#         x = self.cnn(x)
        
#         # Reshape for LSTM (batch, time, features)
#         batch, channels, freq, time = x.size()
#         x = x.permute(0, 3, 2, 1)  # (batch, time, freq, channels)
#         x = x.reshape(batch, time, freq * channels)
        
#         # Expand speaker embedding to match time dimension
#         speaker_embedding = speaker_embedding.unsqueeze(1).expand(-1, time, -1)
        
#         # Concatenate spectrogram features with speaker embedding
#         x = torch.cat([x, speaker_embedding], dim=-1)
        
#         # Process with LSTM
#         x, _ = self.lstm(x)
        
#         # Predict mask
#         mask = self.fc(x)
        
#         # Reshape mask to match original spectrogram dimensions
#         mask = mask.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, freq, time)
#         mask = F.interpolate(mask, size=spectrogram.shape[-2:], mode='bilinear')
        
#         return mask.squeeze(1)

# class SpeakerAwareEnhancer(nn.Module):
#     def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400):
#         super(SpeakerAwareEnhancer, self).__init__()
#         self.sample_rate = sample_rate
        
#         # Load pretrained ECAPA-TDNN speaker encoder
#         self.speaker_encoder = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
#             model_name='ecapa_tdnn'
#         )
        
#         # Freeze speaker encoder
#         for param in self.speaker_encoder.parameters():
#             param.requires_grad = False
#         self.speaker_encoder.eval()
            
#         # VoiceFilter model
#         self.voice_filter = VoiceFilter()
        
#         # STFT transforms
#         self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
#         self.istft = InverseSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        
#         # Create a temporary directory for this session
#         self.temp_dir = tempfile.mkdtemp()
        
#     def __del__(self):
#         # Clean up temporary directory
#         try:
#             shutil.rmtree(self.temp_dir, ignore_errors=True)
#         except:
#             pass
    
#     def _process_single_audio(self, audio_np, sample_idx):
#         """Process a single audio sample and return embedding"""
#         try:
#             # Ensure audio is 1D and properly formatted
#             if len(audio_np.shape) > 1:
#                 audio_np = audio_np.squeeze()
            
#             # Normalize audio to prevent clipping
#             if np.max(np.abs(audio_np)) > 0:
#                 audio_np = audio_np / np.max(np.abs(audio_np))
#             else:
#                 return np.zeros(192, dtype=np.float32)
            
#             # Create temporary file path
#             temp_file = os.path.join(self.temp_dir, f"temp_audio_{sample_idx}.wav")
            
#             # Write audio file
#             sf.write(temp_file, 
#                     audio_np.astype(np.float32), 
#                     self.sample_rate,
#                     subtype='PCM_16')
            
#             # Get embedding
#             embedding = self.speaker_encoder.get_embedding(temp_file)
            
#             # Clean up temporary file
#             os.remove(temp_file)
            
#             # Ensure embedding is numpy array and flatten any extra dimensions
#             if isinstance(embedding, np.ndarray):
#                 embedding = embedding.flatten()
#             else:
#                 embedding = np.array(embedding).flatten()
            
#             # Ensure it's the right size (192 for ECAPA-TDNN)
#             if embedding.shape[0] != 192:
#                 if embedding.shape[0] < 192:
#                     embedding = np.pad(embedding, (0, 192 - embedding.shape[0]))
#                 else:
#                     embedding = embedding[:192]
            
#             return embedding.astype(np.float32)
            
#         except Exception as e:
#             print(f"Warning: Error processing audio sample {sample_idx}: {e}")
#             return np.zeros(192, dtype=np.float32)
        
#     def get_speaker_embeddings(self, reference_audio_batch):
#         """Extract speaker embeddings for a batch of reference audio"""
#         batch_size = reference_audio_batch.size(0)
#         embeddings = []
        
#         # Move to CPU for processing
#         ref_audio_cpu = reference_audio_batch.cpu().numpy()
        
#         with torch.no_grad():
#             for i in range(batch_size):
#                 embedding = self._process_single_audio(ref_audio_cpu[i], i)
#                 embeddings.append(embedding)
        
#         # Convert to tensor and move to correct device
#         embeddings = np.stack(embeddings)
#         embeddings = torch.from_numpy(embeddings).to(reference_audio_batch.device)
        
#         return embeddings
        
#     def forward(self, reference_audio, noisy_audio):
#         # Extract speaker embedding from reference audio
#         speaker_embeddings = self.get_speaker_embeddings(reference_audio)
        
#         # Compute STFT of noisy audio
#         noisy_spec = self.stft(noisy_audio)
        
#         # Handle complex STFT output properly
#         if torch.is_complex(noisy_spec):
#             noisy_mag = torch.abs(noisy_spec)
#         else:
#             noisy_mag = torch.sqrt(noisy_spec.pow(2).sum(-1))
        
#         # Predict mask using VoiceFilter
#         mask = self.voice_filter(noisy_mag, speaker_embeddings)
        
#         # Apply mask to noisy spectrogram
#         if torch.is_complex(noisy_spec):
#             # For complex spectrograms, apply mask to magnitude and preserve phase
#             phase = torch.angle(noisy_spec)
#             masked_magnitude = noisy_mag * mask
#             enhanced_spec = masked_magnitude * torch.exp(1j * phase)
#         else:
#             enhanced_spec = noisy_spec * mask.unsqueeze(-1)
        
#         # Reconstruct enhanced audio
#         enhanced_audio = self.istft(enhanced_spec)
        
#         return enhanced_audio, mask
    
#     def compute_loss(self, enhanced_audio, clean_audio):
#         # Ensure same length
#         min_len = min(enhanced_audio.size(-1), clean_audio.size(-1))
#         enhanced_audio = enhanced_audio[..., :min_len]
#         clean_audio = clean_audio[..., :min_len]
        
#         # Compute STFT of clean audio
#         clean_spec = self.stft(clean_audio)
#         if torch.is_complex(clean_spec):
#             clean_mag = torch.abs(clean_spec)
#         else:
#             clean_mag = torch.sqrt(clean_spec.pow(2).sum(-1))
        
#         # Compute STFT of enhanced audio
#         enhanced_spec = self.stft(enhanced_audio)
#         if torch.is_complex(enhanced_spec):
#             enhanced_mag = torch.abs(enhanced_spec)
#         else:
#             enhanced_mag = torch.sqrt(enhanced_spec.pow(2).sum(-1))
        
#         # Magnitude spectrum loss
#         mag_loss = F.mse_loss(enhanced_mag, clean_mag)
        
#         # Waveform loss
#         waveform_loss = F.mse_loss(enhanced_audio, clean_audio)
        
#         # Combined loss
#         total_loss = mag_loss + 0.1 * waveform_loss
        
#         return total_loss