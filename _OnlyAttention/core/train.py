import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import random
import os
from datetime import datetime
from model import SpeakerAwareEnhancer
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, clean_audio_dir, noise_dir, other_speaker_dir, sample_rate=16000, duration=3.0):
        # Get all WAV files from directories
        self.clean_files = [os.path.join(clean_audio_dir, f) for f in os.listdir(clean_audio_dir) if f.endswith('.wav')]
        self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
        self.other_speaker_files = [os.path.join(other_speaker_dir, f) for f in os.listdir(other_speaker_dir) if f.endswith('.wav')]
        
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples_per_audio = int(sample_rate * duration)
        
        print(f"Dataset initialized:")
        print(f"  Clean audio files: {len(self.clean_files)}")
        print(f"  Noise files: {len(self.noise_files)}")
        print(f"  Other speaker files: {len(self.other_speaker_files)}")
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load clean audio
        clean_audio, _ = sf.read(self.clean_files[idx])
        clean_audio = self._preprocess_audio(clean_audio)
        
        # Create reference audio (random segment from clean audio)
        ref_start = random.randint(0, len(clean_audio) - self.samples_per_audio)
        reference_audio = clean_audio[ref_start:ref_start + self.samples_per_audio]
        
        # Create noisy audio by mixing with noise and other speakers
        noisy_audio = clean_audio.copy()
        
        # Add random noise
        if random.random() > 0.3 and len(self.noise_files) > 0:
            noise_file = random.choice(self.noise_files)
            noise, _ = sf.read(noise_file)
            noise = self._preprocess_audio(noise)
            noise_gain = random.uniform(0.1, 0.5)
            noisy_audio = noisy_audio + noise_gain * noise[:len(noisy_audio)]
        
        # Add other speakers
        if random.random() > 0.5 and len(self.other_speaker_files) > 0:
            other_speaker_file = random.choice(self.other_speaker_files)
            other_speaker, _ = sf.read(other_speaker_file)
            other_speaker = self._preprocess_audio(other_speaker)
            speaker_gain = random.uniform(0.1, 0.3)
            noisy_audio = noisy_audio + speaker_gain * other_speaker[:len(noisy_audio)]
        
        # Normalize to prevent clipping
        noisy_audio = noisy_audio / (np.max(np.abs(noisy_audio)) + 1e-7)
        
        # Random segment for training
        start = random.randint(0, len(clean_audio) - self.samples_per_audio)
        clean_audio = clean_audio[start:start + self.samples_per_audio]
        noisy_audio = noisy_audio[start:start + self.samples_per_audio]
        
        return {
            'reference_audio': torch.FloatTensor(reference_audio),
            'noisy_audio': torch.FloatTensor(noisy_audio),
            'clean_audio': torch.FloatTensor(clean_audio)
        }
    
    def _preprocess_audio(self, audio):
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-7)
        
        # Pad or truncate to desired length
        if len(audio) < self.samples_per_audio:
            audio = np.pad(audio, (0, self.samples_per_audio - len(audio)))
        else:
            audio = audio[:self.samples_per_audio]
        
        return audio

def train_model(clean_dir, noise_dir, other_speaker_dir, checkpoint_dir="checkpoints", log_file="training_log.txt"):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SpeakerAwareEnhancer().to(device)
    print("Model loaded successfully")
    
    # Create dataset and dataloader
    dataset = AudioDataset(clean_dir, noise_dir, other_speaker_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    print(f"DataLoader created with {len(dataset)} samples, {len(dataloader)} batches")
    
    # Optimizer
    optimizer = optim.Adam(model.voice_filter.parameters(), lr=1e-4)
    
    # Initialize log file with header
    with open(log_file, 'w') as f:
        f.write(f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        f.write("Epoch,Average_Loss,Min_Batch_Loss,Max_Batch_Loss,Training_Time(min)\n")
    
    # Training loop
    num_epochs = 25
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Loss log will be written to: {log_file}")
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        model.train()
        total_loss = 0
        batch_losses = []
        
        # Create progress bar for batches
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                   unit='batch', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                reference_audio = batch['reference_audio'].to(device)
                noisy_audio = batch['noisy_audio'].to(device)
                clean_audio = batch['clean_audio'].to(device)
                
                # Forward pass
                enhanced_audio, _ = model(reference_audio, noisy_audio)
                
                # Compute loss
                loss = model.compute_loss(enhanced_audio, clean_audio)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_value = loss.item()
                total_loss += loss_value
                batch_losses.append(loss_value)
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 
                                'Current': f'{loss_value:.4f}'})
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch statistics
        avg_epoch_loss = total_loss / len(dataloader)
        min_batch_loss = min(batch_losses) if batch_losses else 0
        max_batch_loss = max(batch_losses) if batch_losses else 0
        epoch_time = (datetime.now() - epoch_start_time).total_seconds() / 60
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} completed - Average Loss: {avg_epoch_loss:.4f} "
              f"(Min: {min_batch_loss:.4f}, Max: {max_batch_loss:.4f}) "
              f"- Time: {epoch_time:.2f} min")
        
        # Save model checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"speaker_enhancer_epoch_{epoch+1:03d}.pt")
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'min_batch_loss': min_batch_loss,
            'max_batch_loss': max_batch_loss,
            'training_time_minutes': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Write loss to log file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_epoch_loss:.6f},{min_batch_loss:.6f},"
                   f"{max_batch_loss:.6f},{epoch_time:.2f}\n")
        
        # Also save latest checkpoint (overwrites each epoch)
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint_data, latest_checkpoint_path)
    
    # Save final summary to log file
    with open(log_file, 'a') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {num_epochs}\n")
        f.write(f"Final loss: {avg_epoch_loss:.6f}\n")
    
    print(f"\nTraining completed! Final checkpoint and logs saved.")
    print(f"All checkpoints are available in: {checkpoint_dir}")
    print(f"Training log saved to: {log_file}")

def resume_training(checkpoint_path, clean_dir, noise_dir, other_speaker_dir, 
                   additional_epochs=10, checkpoint_dir="checkpoints", log_file="training_log.txt"):
    """Resume training from a checkpoint"""
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Resuming training from epoch {checkpoint['epoch']}")
    
    # Initialize model and load state
    model = SpeakerAwareEnhancer().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize optimizer and load state
    optimizer = optim.Adam(model.voice_filter.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Create dataset and dataloader
    dataset = AudioDataset(clean_dir, noise_dir, other_speaker_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    start_epoch = checkpoint['epoch']
    
    # Append to existing log file
    with open(log_file, 'a') as f:
        f.write(f"\n--- Resumed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    
    # Continue training
    for epoch in range(start_epoch, start_epoch + additional_epochs):
        epoch_start_time = datetime.now()
        model.train()
        total_loss = 0
        batch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{start_epoch + additional_epochs}', 
                   unit='batch', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                reference_audio = batch['reference_audio'].to(device)
                noisy_audio = batch['noisy_audio'].to(device)
                clean_audio = batch['clean_audio'].to(device)
                
                enhanced_audio, _ = model(reference_audio, noisy_audio)
                loss = model.compute_loss(enhanced_audio, clean_audio)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_value = loss.item()
                total_loss += loss_value
                batch_losses.append(loss_value)
                
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 
                                'Current': f'{loss_value:.4f}'})
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        # Save checkpoint and log
        avg_epoch_loss = total_loss / len(dataloader)
        min_batch_loss = min(batch_losses) if batch_losses else 0
        max_batch_loss = max(batch_losses) if batch_losses else 0
        epoch_time = (datetime.now() - epoch_start_time).total_seconds() / 60
        
        checkpoint_path = os.path.join(checkpoint_dir, f"speaker_enhancer_epoch_{epoch+1:03d}.pt")
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'min_batch_loss': min_batch_loss,
            'max_batch_loss': max_batch_loss,
            'training_time_minutes': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_epoch_loss:.6f},{min_batch_loss:.6f},"
                   f"{max_batch_loss:.6f},{epoch_time:.2f}\n")
        
        print(f"Epoch {epoch+1} completed - Loss: {avg_epoch_loss:.4f} - Time: {epoch_time:.2f} min")

if __name__ == "__main__":
    # Replace these paths with your actual directory paths
    clean_audio_dir = "/home/mayank.500106701/Audio_Cleaning_Major1/data/clean_audio/small_data"
    noise_dir = "/home/mayank.500106701/Audio_Cleaning_Major1/data/noise_audio/small_data"
    other_speaker_dir = "/home/mayank.500106701/Audio_Cleaning_Major1/data/noise_audio/small_data"
    
    # Check if directories exist
    for dir_path, dir_name in [(clean_audio_dir, "clean_audio"), 
                               (noise_dir, "noise_audio"), 
                               (other_speaker_dir, "other_speaker")]:
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_name} directory '{dir_path}' does not exist")
        else:
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            print(f"{dir_name} directory: {len(wav_files)} WAV files found")
    
    # Start fresh training
    # train_model(clean_audio_dir, noise_dir, other_speaker_dir)
    
    # Example of how to resume training:
    resume_training("/home/mayank.500106701/Audio_Cleaning_Major1/OnlyAttention/checkpoints/speaker_enhancer_epoch_025.pt", 
                   clean_audio_dir, noise_dir, other_speaker_dir,
                   additional_epochs=25)
