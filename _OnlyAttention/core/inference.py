import torch
import soundfile as sf
import numpy as np
import argparse
import os
from datetime import datetime
from model import SpeakerAwareEnhancer
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import warnings
warnings.filterwarnings('ignore')

class VoiceEnhancementInference:
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize the inference system with a trained checkpoint
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.model = SpeakerAwareEnhancer().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Checkpoint info:")
        print(f"  - Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  - Training loss: {checkpoint.get('loss', 'Unknown'):.6f}")
        print(f"  - Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        
    def load_audio(self, audio_path, target_sample_rate=16000):
        """
        Load and preprocess audio file
        
        Args:
            audio_path (str): Path to audio file
            target_sample_rate (int): Target sample rate for processing
            
        Returns:
            numpy.ndarray: Preprocessed audio
        """
        try:
            audio, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed (simple resampling)
            if sample_rate != target_sample_rate:
                print(f"Warning: Audio sample rate is {sample_rate}Hz, expected {target_sample_rate}Hz")
                print("Consider using proper resampling tools for better quality")
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32)
            
        except Exception as e:
            raise Exception(f"Error loading audio file {audio_path}: {e}")
    
    def enhance_audio(self, reference_audio_path, noisy_audio_path, output_path=None, 
                     visualize=False, chunk_duration=3.0):
        """
        Enhance noisy audio using reference speaker audio
        
        Args:
            reference_audio_path (str): Path to clean reference audio of target speaker
            noisy_audio_path (str): Path to noisy audio to enhance
            output_path (str): Path to save enhanced audio (optional)
            visualize (bool): Whether to create spectrograms for visualization
            chunk_duration (float): Duration of audio chunks for processing (seconds)
            
        Returns:
            tuple: (enhanced_audio_array, sample_rate)
        """
        print("\n" + "="*60)
        print("VOICE ENHANCEMENT INFERENCE")
        print("="*60)
        
        # Load audio files
        print("Loading audio files...")
        reference_audio = self.load_audio(reference_audio_path)
        noisy_audio = self.load_audio(noisy_audio_path)
        
        print(f"Reference audio duration: {len(reference_audio)/16000:.2f}s")
        print(f"Noisy audio duration: {len(noisy_audio)/16000:.2f}s")
        
        # Process audio in chunks if it's too long
        sample_rate = 16000
        chunk_samples = int(chunk_duration * sample_rate)
        
        if len(noisy_audio) <= chunk_samples:
            # Process single chunk
            enhanced_audio = self._process_chunk(reference_audio, noisy_audio)
        else:
            # Process in overlapping chunks
            enhanced_audio = self._process_long_audio(reference_audio, noisy_audio, 
                                                    chunk_samples)
        
        # Save enhanced audio
        if output_path:
            sf.write(output_path, enhanced_audio, sample_rate)
            print(f"Enhanced audio saved to: {output_path}")
        else:
            # Generate automatic filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"enhanced_audio_{timestamp}.wav"
            sf.write(output_path, enhanced_audio, sample_rate)
            print(f"Enhanced audio saved to: {output_path}")
        
        # Create visualizations
        if visualize:
            self._create_visualization(noisy_audio, enhanced_audio, sample_rate, 
                                     output_path.replace('.wav', '_comparison.png'))
        
        print("Enhancement completed successfully!")
        return enhanced_audio, sample_rate
    
    def _process_chunk(self, reference_audio, noisy_audio):
        """Process a single chunk of audio"""
        with torch.no_grad():
            # Ensure minimum length for reference
            min_ref_length = 16000  # 1 second minimum
            if len(reference_audio) < min_ref_length:
                # Repeat reference audio if too short
                repeat_factor = int(np.ceil(min_ref_length / len(reference_audio)))
                reference_audio = np.tile(reference_audio, repeat_factor)[:min_ref_length]
            
            # Take a segment of reference audio
            ref_segment_length = min(48000, len(reference_audio))  # Up to 3 seconds
            reference_segment = reference_audio[:ref_segment_length]
            
            # Convert to tensors
            ref_tensor = torch.FloatTensor(reference_segment).unsqueeze(0).to(self.device)
            noisy_tensor = torch.FloatTensor(noisy_audio).unsqueeze(0).to(self.device)
            
            # Enhance audio
            enhanced_tensor, mask = self.model(ref_tensor, noisy_tensor)
            
            # Convert back to numpy
            enhanced_audio = enhanced_tensor.squeeze().cpu().numpy()
            
            return enhanced_audio
    
    def _process_long_audio(self, reference_audio, noisy_audio, chunk_samples, overlap=0.25):
        """Process long audio in overlapping chunks"""
        print("Processing long audio in chunks...")
        
        overlap_samples = int(chunk_samples * overlap)
        hop_samples = chunk_samples - overlap_samples
        
        enhanced_chunks = []
        num_chunks = int(np.ceil(len(noisy_audio) / hop_samples))
        
        for i in range(num_chunks):
            start_idx = i * hop_samples
            end_idx = min(start_idx + chunk_samples, len(noisy_audio))
            
            # Get chunk
            chunk = noisy_audio[start_idx:end_idx]
            
            # Pad if necessary
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            
            # Process chunk
            enhanced_chunk = self._process_chunk(reference_audio, chunk)
            
            # Handle overlapping for seamless reconstruction
            if i == 0:
                # First chunk
                enhanced_chunks.append(enhanced_chunk[:hop_samples])
            elif i == num_chunks - 1:
                # Last chunk
                enhanced_chunks.append(enhanced_chunk[overlap_samples:len(chunk)])
            else:
                # Middle chunk
                enhanced_chunks.append(enhanced_chunk[overlap_samples:hop_samples + overlap_samples])
            
            print(f"Processed chunk {i+1}/{num_chunks}")
        
        # Concatenate all chunks
        enhanced_audio = np.concatenate(enhanced_chunks)
        
        # Trim to original length
        enhanced_audio = enhanced_audio[:len(noisy_audio)]
        
        return enhanced_audio
    
    def _create_visualization(self, noisy_audio, enhanced_audio, sample_rate, save_path):
        """Create spectrogram comparison"""
        print("Creating visualization...")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Compute spectrograms
        f1, t1, Sxx1 = spectrogram(noisy_audio, sample_rate, nperseg=512, noverlap=256)
        f2, t2, Sxx2 = spectrogram(enhanced_audio, sample_rate, nperseg=512, noverlap=256)
        
        # Plot noisy spectrogram
        im1 = axes[0].pcolormesh(t1, f1, 10*np.log10(Sxx1 + 1e-10), shading='gouraud')
        axes[0].set_title('Noisy Audio Spectrogram')
        axes[0].set_ylabel('Frequency [Hz]')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot enhanced spectrogram
        im2 = axes[1].pcolormesh(t2, f2, 10*np.log10(Sxx2 + 1e-10), shading='gouraud')
        axes[1].set_title('Enhanced Audio Spectrogram')
        axes[1].set_ylabel('Frequency [Hz]')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot waveform comparison
        time_axis = np.linspace(0, len(noisy_audio)/sample_rate, len(noisy_audio))
        axes[2].plot(time_axis, noisy_audio, alpha=0.7, label='Noisy', color='red')
        axes[2].plot(time_axis, enhanced_audio, alpha=0.7, label='Enhanced', color='blue')
        axes[2].set_title('Waveform Comparison')
        axes[2].set_xlabel('Time [s]')
        axes[2].set_ylabel('Amplitude')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Voice Enhancement Inference')
    parser.add_argument('--checkpoint', '-c', required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--reference', '-r', required=True,
                       help='Path to reference audio file (clean speech of target speaker)')
    parser.add_argument('--noisy', '-n', required=True,
                       help='Path to noisy audio file to enhance')
    parser.add_argument('--output', '-o', default=None,
                       help='Output path for enhanced audio (auto-generated if not specified)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Create spectrogram visualization')
    parser.add_argument('--device', '-d', choices=['cpu', 'cuda'], default=None,
                       help='Device to use for inference')
    parser.add_argument('--chunk-duration', '-cd', type=float, default=3.0,
                       help='Duration of audio chunks for processing (seconds)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.reference):
        print(f"Error: Reference audio file not found: {args.reference}")
        return
    
    if not os.path.exists(args.noisy):
        print(f"Error: Noisy audio file not found: {args.noisy}")
        return
    
    try:
        # Initialize inference system
        enhancer = VoiceEnhancementInference(args.checkpoint, args.device)
        
        # Enhance audio
        enhanced_audio, sample_rate = enhancer.enhance_audio(
            reference_audio_path=args.reference,
            noisy_audio_path=args.noisy,
            output_path=args.output,
            visualize=args.visualize,
            chunk_duration=args.chunk_duration
        )
        
        print(f"\nResults:")
        print(f"Enhanced audio shape: {enhanced_audio.shape}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(enhanced_audio)/sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()