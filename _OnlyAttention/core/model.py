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
# import math


# class PositionalEncoding(nn.Module):
#     """Relative positional encoding for streaming"""
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         # x: (batch, seq_len, d_model)
#         return x + self.pe[:x.size(1), :].unsqueeze(0)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1, look_back=None):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0
        
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.look_back = look_back
        
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.out_linear = nn.Linear(d_model, d_model)
        
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
        
#         # Linear projections
#         Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
#         # Attention scores
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
#         # Apply look-back masking for streaming (causal mask)
#         if self.look_back is not None and mask is None:
#             seq_len = query.size(1)
#             mask = torch.ones(seq_len, seq_len, device=query.device)
#             for i in range(seq_len):
#                 start = max(0, i - self.look_back + 1)
#                 mask[i, :start] = 0
#             mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
        
#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
        
#         # Apply attention to values
#         context = torch.matmul(attn, V)
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
#         return self.out_linear(context)


# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff=1024, dropout=0.1):
#         super(FeedForward, self).__init__()
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ff, d_model)
    
#     def forward(self, x):
#         return self.linear2(self.dropout(F.relu(self.linear1(x))))


# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff=1024, dropout=0.1, look_back=100):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, look_back)
#         self.feed_forward = FeedForward(d_model, d_ff, dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         # Self-attention with residual connection
#         attn_output = self.self_attn(x, x, x)
#         x = self.norm1(x + self.dropout(attn_output))
        
#         # Feed-forward with residual connection
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
        
#         return x


# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff=1024, dropout=0.1, look_back=100):
#         super(TransformerDecoderLayer, self).__init__()
#         self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, look_back=None)
#         self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, look_back)
#         self.feed_forward = FeedForward(d_model, d_ff, dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x, encoder_output):
#         # Cross-attention (Q from decoder, K,V from encoder/enrollment)
#         cross_output = self.cross_attn(x, encoder_output, encoder_output)
#         x = self.norm1(x + self.dropout(cross_output))
        
#         # Self-attention
#         self_output = self.self_attn(x, x, x)
#         x = self.norm2(x + self.dropout(self_output))
        
#         # Feed-forward
#         ff_output = self.feed_forward(x)
#         x = self.norm3(x + self.dropout(ff_output))
        
#         return x


# class VoiceFilter(nn.Module):
#     """Transformer-based VoiceFilter with Cross-Attention"""
#     def __init__(self, freq_dim=257, embedding_dim=192, d_model=256, 
#                  num_encoder_layers=3, num_decoder_layers=3, 
#                  num_heads=8, d_ff=1024, dropout=0.1, look_back=100):
#         super(VoiceFilter, self).__init__()
        
#         self.freq_dim = freq_dim
#         self.d_model = d_model
        
#         # Input projection: map frequency bins to model dimension
#         self.input_projection = nn.Linear(freq_dim, d_model)
        
#         # Embedding projection: map speaker embedding to model dimension
#         self.embedding_projection = nn.Linear(embedding_dim, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Encoder layers (process noisy spectrogram)
#         self.encoder_layers = nn.ModuleList([
#             TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, look_back)
#             for _ in range(num_encoder_layers)
#         ])
        
#         # Decoder layers (cross-attention with speaker embedding)
#         self.decoder_layers = nn.ModuleList([
#             TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, look_back)
#             for _ in range(num_decoder_layers)
#         ])
        
#         # Output projection: map back to frequency dimension for mask
#         self.output_projection = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model // 2, freq_dim),
#             nn.Sigmoid()
#         )
        
#     def forward(self, spectrogram, speaker_embedding):
#         """
#         Args:
#             spectrogram: (batch, freq, time) - magnitude spectrogram
#             speaker_embedding: (batch, embedding_dim) or (batch, seq_len, embedding_dim)
#         Returns:
#             mask: (batch, freq, time) - predicted mask
#         """
#         batch_size, freq, time = spectrogram.shape
        
#         # Transpose to (batch, time, freq) for transformer processing
#         x = spectrogram.permute(0, 2, 1)  # (batch, time, freq)
        
#         # Project to model dimension
#         x = self.input_projection(x)  # (batch, time, d_model)
#         x = self.pos_encoder(x)
        
#         # Encoder: process noisy spectrogram
#         for layer in self.encoder_layers:
#             x = layer(x)
        
#         # Process speaker embedding
#         if len(speaker_embedding.shape) == 2:
#             # Single vector: (batch, embedding_dim) -> (batch, 1, embedding_dim)
#             speaker_embedding = speaker_embedding.unsqueeze(1)
        
#         # Project speaker embedding to model dimension
#         speaker_emb = self.embedding_projection(speaker_embedding)  # (batch, seq_len, d_model)
        
#         # If single embedding, expand to match time dimension for better cross-attention
#         if speaker_emb.size(1) == 1:
#             # Optionally expand single embedding to multiple time steps
#             # This allows the model to attend to the same embedding from different query positions
#             speaker_emb = speaker_emb.expand(-1, min(10, time), -1)  # Expand to 10 steps
        
#         # Decoder: cross-attention with speaker embedding
#         for layer in self.decoder_layers:
#             x = layer(x, speaker_emb)
        
#         # Generate mask
#         mask = self.output_projection(x)  # (batch, time, freq)
        
#         # Transpose back to (batch, freq, time)
#         mask = mask.permute(0, 2, 1)
        
#         return mask


# class SpeakerAwareEnhancer(nn.Module):
#     def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400,
#                  d_model=256, num_encoder_layers=3, num_decoder_layers=3, 
#                  num_heads=8, d_ff=1024, dropout=0.1, look_back=100):
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
            
#         # Transformer-based VoiceFilter with cross-attention
#         freq_dim = n_fft // 2 + 1  # 257 for n_fft=512
#         self.voice_filter = VoiceFilter(
#             freq_dim=freq_dim,
#             embedding_dim=192,
#             d_model=d_model,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             num_heads=num_heads,
#             d_ff=d_ff,
#             dropout=dropout,
#             look_back=look_back
#         )
        
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
#             # Ensure we have a proper numpy array
#             if torch.is_tensor(audio_np):
#                 if audio_np.is_cuda:
#                     audio_np = audio_np.detach().cpu().numpy()
#                 else:
#                     audio_np = audio_np.detach().numpy()
            
#             # Ensure audio is 1D and properly formatted
#             if len(audio_np.shape) > 1:
#                 audio_np = audio_np.squeeze()
            
#             # Convert to proper numpy array if needed
#             audio_np = np.asarray(audio_np, dtype=np.float32)
            
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
#             print(f"Audio type: {type(audio_np)}, Audio shape: {getattr(audio_np, 'shape', 'No shape')}")
#             if torch.is_tensor(audio_np):
#                 print(f"Tensor device: {audio_np.device}")
#             return np.zeros(192, dtype=np.float32)
        
#     def get_speaker_embeddings(self, reference_audio_batch):
#         """Extract speaker embeddings for a batch of reference audio"""
#         batch_size = reference_audio_batch.size(0)
#         embeddings = []
#         original_device = reference_audio_batch.device
        
#         with torch.no_grad():
#             # Move to CPU and ensure it's properly detached from any gradients
#             if reference_audio_batch.is_cuda:
#                 ref_audio_cpu = reference_audio_batch.detach().cpu().clone().numpy()
#             else:
#                 ref_audio_cpu = reference_audio_batch.detach().clone().numpy()
            
#             for i in range(batch_size):
#                 # Make sure we're working with a clean numpy array
#                 audio_sample = ref_audio_cpu[i].copy()
#                 embedding = self._process_single_audio(audio_sample, i)
#                 embeddings.append(embedding)
        
#         # Convert to tensor and move back to original device
#         embeddings = np.stack(embeddings)
#         embeddings = torch.from_numpy(embeddings).to(original_device)
        
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
        
#         # Predict mask using Transformer-based VoiceFilter
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





# gpu fix

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import STFT, InverseSTFT
import nemo.collections.asr as nemo_asr
import tempfile
import soundfile as sf
import numpy as np
import os
import shutil
import math


class PositionalEncoding(nn.Module):
    """Relative positional encoding for streaming"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, look_back=None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.look_back = look_back
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply look-back masking for streaming (causal mask)
        if self.look_back is not None and mask is None:
            seq_len = query.size(1)
            mask = torch.ones(seq_len, seq_len, device=query.device)
            for i in range(seq_len):
                start = max(0, i - self.look_back + 1)
                mask[i, :start] = 0
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=1024, dropout=0.1, look_back=100):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, look_back)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=1024, dropout=0.1, look_back=100):
        super(TransformerDecoderLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, look_back=None)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, look_back)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output):
        # Cross-attention (Q from decoder, K,V from encoder/enrollment)
        cross_output = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm1(x + self.dropout(cross_output))
        
        # Self-attention
        self_output = self.self_attn(x, x, x)
        x = self.norm2(x + self.dropout(self_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class VoiceFilter(nn.Module):
    """Transformer-based VoiceFilter with Cross-Attention"""
    def __init__(self, freq_dim=257, embedding_dim=192, d_model=256, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 num_heads=8, d_ff=1024, dropout=0.1, look_back=100):
        super(VoiceFilter, self).__init__()
        
        self.freq_dim = freq_dim
        self.d_model = d_model
        
        # Input projection: map frequency bins to model dimension
        self.input_projection = nn.Linear(freq_dim, d_model)
        
        # Embedding projection: map speaker embedding to model dimension
        self.embedding_projection = nn.Linear(embedding_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder layers (process noisy spectrogram)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, look_back)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers (cross-attention with speaker embedding)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, look_back)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection: map back to frequency dimension for mask
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, freq_dim),
            nn.Sigmoid()
        )
        
    def forward(self, spectrogram, speaker_embedding):
        """
        Args:
            spectrogram: (batch, freq, time) - magnitude spectrogram
            speaker_embedding: (batch, embedding_dim) or (batch, seq_len, embedding_dim)
        Returns:
            mask: (batch, freq, time) - predicted mask
        """
        batch_size, freq, time = spectrogram.shape
        
        # Transpose to (batch, time, freq) for transformer processing
        x = spectrogram.permute(0, 2, 1)  # (batch, time, freq)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, time, d_model)
        x = self.pos_encoder(x)
        
        # Encoder: process noisy spectrogram
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Process speaker embedding
        if len(speaker_embedding.shape) == 2:
            # Single vector: (batch, embedding_dim) -> (batch, 1, embedding_dim)
            speaker_embedding = speaker_embedding.unsqueeze(1)
        
        # Project speaker embedding to model dimension
        speaker_emb = self.embedding_projection(speaker_embedding)  # (batch, seq_len, d_model)
        
        # If single embedding, expand to match time dimension for better cross-attention
        if speaker_emb.size(1) == 1:
            # Optionally expand single embedding to multiple time steps
            # This allows the model to attend to the same embedding from different query positions
            speaker_emb = speaker_emb.expand(-1, min(10, time), -1)  # Expand to 10 steps
        
        # Decoder: cross-attention with speaker embedding
        for layer in self.decoder_layers:
            x = layer(x, speaker_emb)
        
        # Generate mask
        mask = self.output_projection(x)  # (batch, time, freq)
        
        # Transpose back to (batch, freq, time)
        mask = mask.permute(0, 2, 1)
        
        return mask


class SpeakerAwareEnhancer(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400,
                 d_model=256, num_encoder_layers=3, num_decoder_layers=3, 
                 num_heads=8, d_ff=1024, dropout=0.1, look_back=100):
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
            
        # Transformer-based VoiceFilter with cross-attention
        freq_dim = n_fft // 2 + 1  # 257 for n_fft=512
        self.voice_filter = VoiceFilter(
            freq_dim=freq_dim,
            embedding_dim=192,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            look_back=look_back
        )
        
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
            # CRITICAL FIX: Ensure audio is completely on CPU and converted to numpy
            if torch.is_tensor(audio_np):
                audio_np = audio_np.detach().cpu().numpy()
            
            # Ensure it's a proper numpy array (not a view or memmap)
            audio_np = np.array(audio_np, dtype=np.float32)
            
            # Ensure audio is 1D
            if len(audio_np.shape) > 1:
                audio_np = audio_np.squeeze()
            
            # Ensure 1D array
            if len(audio_np.shape) == 0:
                return np.zeros(192, dtype=np.float32)
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val
            else:
                return np.zeros(192, dtype=np.float32)
            
            # Create temporary file path
            temp_file = os.path.join(self.temp_dir, f"temp_audio_{sample_idx}.wav")
            
            # Write audio file
            sf.write(temp_file, audio_np, self.sample_rate, subtype='PCM_16')
            
            # Get embedding
            embedding = self.speaker_encoder.get_embedding(temp_file)
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Process embedding
            if torch.is_tensor(embedding):
                embedding = embedding.detach().cpu().numpy()
            
            # Ensure it's a numpy array and flatten
            embedding = np.array(embedding, dtype=np.float32).flatten()
            
            # Ensure correct size (192 for ECAPA-TDNN)
            if embedding.shape[0] != 192:
                if embedding.shape[0] < 192:
                    embedding = np.pad(embedding, (0, 192 - embedding.shape[0]))
                else:
                    embedding = embedding[:192]
            
            return embedding
            
        except Exception as e:
            print(f"Warning: Error processing audio sample {sample_idx}: {e}")
            print(f"Audio type: {type(audio_np)}")
            if hasattr(audio_np, 'shape'):
                print(f"Audio shape: {audio_np.shape}")
            if torch.is_tensor(audio_np):
                print(f"Tensor device: {audio_np.device}")
            return np.zeros(192, dtype=np.float32)
        
    def get_speaker_embeddings(self, reference_audio_batch):
        """Extract speaker embeddings for a batch of reference audio"""
        batch_size = reference_audio_batch.size(0)
        embeddings = []
        original_device = reference_audio_batch.device
        
        with torch.no_grad():
            # CRITICAL FIX: Move entire batch to CPU first, then convert to numpy
            reference_audio_cpu = reference_audio_batch.detach().cpu().numpy()
            
            # Process each sample
            for i in range(batch_size):
                # Get audio sample as numpy array
                audio_sample = reference_audio_cpu[i]
                
                # Process and get embedding
                embedding = self._process_single_audio(audio_sample, i)
                embeddings.append(embedding)
        
        # Stack embeddings and move back to original device
        embeddings = np.stack(embeddings)
        embeddings_tensor = torch.from_numpy(embeddings).to(original_device)
        
        return embeddings_tensor
        
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
        
        # Predict mask using Transformer-based VoiceFilter
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