import torch
import numpy as np
import pyaudio
import threading
import queue
import argparse
import os
import soundfile as sf
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.model import SpeakerAwareEnhancer
import warnings
warnings.filterwarnings('ignore')

class RealTimeVoiceEnhancer:
    def __init__(self, checkpoint_path, reference_audio_path, device=None):
        """
        Initialize real-time voice enhancer
        
        Args:
            checkpoint_path (str): Path to trained model checkpoint
            reference_audio_path (str): Path to reference audio of your voice
            device (str): Device to run inference on
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # Process 500ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_ratio = 0.25  # 25% overlap between chunks
        self.overlap_size = int(self.chunk_size * self.overlap_ratio)
        
        # Load model
        self._load_model(checkpoint_path)
        
        # Load and prepare reference audio
        self._load_reference_audio(reference_audio_path)
        
        # Audio streaming setup
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        
        # Threading and queues
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Buffer for overlapping
        self.input_buffer = deque(maxlen=self.chunk_size + self.overlap_size)
        self.output_buffer = deque(maxlen=self.chunk_size)
        
        # Statistics
        self.processing_times = deque(maxlen=100)
        self.total_chunks_processed = 0
        
        # Visualization setup
        self.visualize = False
        self.fig = None
        self.ax = None
        self.line_input = None
        self.line_output = None
        
    def _load_model(self, checkpoint_path):
        """Load the trained model"""
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = SpeakerAwareEnhancer().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully (Epoch {checkpoint.get('epoch', 'Unknown')})")
    
    def _load_reference_audio(self, reference_path):
        """Load and prepare reference audio"""
        print(f"Loading reference audio from: {reference_path}")
        
        reference_audio, sr = sf.read(reference_path)
        
        # Convert to mono if needed
        if len(reference_audio.shape) > 1:
            reference_audio = np.mean(reference_audio, axis=1)
        
        # Resample if needed (basic resampling)
        if sr != self.sample_rate:
            print(f"Warning: Reference audio sample rate is {sr}Hz, expected {self.sample_rate}Hz")
        
        # Normalize
        if np.max(np.abs(reference_audio)) > 0:
            reference_audio = reference_audio / np.max(np.abs(reference_audio))
        
        # Ensure minimum length (3 seconds)
        min_length = self.sample_rate * 3
        if len(reference_audio) < min_length:
            repeat_factor = int(np.ceil(min_length / len(reference_audio)))
            reference_audio = np.tile(reference_audio, repeat_factor)[:min_length]
        
        # Store as tensor
        self.reference_tensor = torch.FloatTensor(reference_audio).unsqueeze(0).to(self.device)
        print(f"Reference audio prepared: {len(reference_audio)/self.sample_rate:.2f}s duration")
    
    def _audio_callback_input(self, in_data, frame_count, time_info, status):
        """Callback for audio input"""
        if status:
            print(f"Input callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to queue if not full
        try:
            self.input_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Skip if queue is full
        
        return (None, pyaudio.paContinue)
    
    def _audio_callback_output(self, in_data, frame_count, time_info, status):
        """Callback for audio output"""
        if status:
            print(f"Output callback status: {status}")
        
        try:
            # Get processed audio from queue
            audio_data = self.output_queue.get_nowait()
            
            # Ensure correct length
            if len(audio_data) != frame_count:
                if len(audio_data) < frame_count:
                    audio_data = np.pad(audio_data, (0, frame_count - len(audio_data)))
                else:
                    audio_data = audio_data[:frame_count]
            
            return (audio_data.astype(np.float32).tobytes(), pyaudio.paContinue)
            
        except queue.Empty:
            # Return silence if no processed audio available
            silence = np.zeros(frame_count, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)
    
    def _process_audio_chunk(self, audio_chunk):
        """Process a single audio chunk through the model"""
        start_time = time.time()
        
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.FloatTensor(audio_chunk).unsqueeze(0).to(self.device)
            
            # Enhance audio
            enhanced_tensor, _ = self.model(self.reference_tensor, input_tensor)
            
            # Convert back to numpy
            enhanced_audio = enhanced_tensor.squeeze().cpu().numpy()
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return enhanced_audio
    
    def _processing_thread(self):
        """Main processing thread"""
        print("Processing thread started")
        
        while self.running:
            try:
                # Get input audio
                input_chunk = self.input_queue.get(timeout=0.1)
                
                # Add to input buffer
                self.input_buffer.extend(input_chunk)
                
                # Process when we have enough data
                if len(self.input_buffer) >= self.chunk_size:
                    # Get chunk for processing
                    chunk_data = np.array(list(self.input_buffer)[:self.chunk_size])
                    
                    # Remove processed samples (with overlap)
                    for _ in range(self.chunk_size - self.overlap_size):
                        if self.input_buffer:
                            self.input_buffer.popleft()
                    
                    # Process chunk
                    try:
                        enhanced_chunk = self._process_audio_chunk(chunk_data)
                        self.total_chunks_processed += 1
                        
                        # Add to output queue
                        output_chunk = enhanced_chunk[:self.chunk_size - self.overlap_size]
                        try:
                            self.output_queue.put_nowait(output_chunk)
                        except queue.Full:
                            # Remove oldest if queue is full
                            try:
                                self.output_queue.get_nowait()
                                self.output_queue.put_nowait(output_chunk)
                            except queue.Empty:
                                pass
                                
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                        # Pass through original audio on error
                        try:
                            self.output_queue.put_nowait(chunk_data[:self.chunk_size - self.overlap_size])
                        except queue.Full:
                            pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                break
        
        print("Processing thread stopped")
    
    def _setup_visualization(self):
        """Setup real-time visualization"""
        if not self.visualize:
            return
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Initialize empty plots
        self.line_input, = self.ax1.plot([], [], 'r-', label='Input (Noisy)', alpha=0.7)
        self.line_output, = self.ax2.plot([], [], 'b-', label='Output (Enhanced)', alpha=0.7)
        
        self.ax1.set_title('Real-time Audio Processing')
        self.ax1.set_ylabel('Input Amplitude')
        self.ax1.set_ylim(-1, 1)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_ylabel('Output Amplitude')
        self.ax2.set_xlabel('Sample')
        self.ax2.set_ylim(-1, 1)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _update_visualization(self, input_data, output_data):
        """Update real-time visualization"""
        if not self.visualize or self.fig is None:
            return
        
        try:
            # Update input plot
            x = np.arange(len(input_data))
            self.line_input.set_data(x, input_data)
            self.ax1.set_xlim(0, len(input_data))
            
            # Update output plot
            x = np.arange(len(output_data))
            self.line_output.set_data(x, output_data)
            self.ax2.set_xlim(0, len(output_data))
            
            # Refresh plots
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def start_enhancement(self, input_device=None, output_device=None, visualize=False, 
                         save_output=False, output_file="realtime_enhanced.wav"):
        """Start real-time enhancement"""
        self.visualize = visualize
        
        print("\n" + "="*60)
        print("REAL-TIME VOICE ENHANCEMENT")
        print("="*60)
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk size: {self.chunk_size} samples ({self.chunk_duration}s)")
        print(f"Overlap: {self.overlap_size} samples ({self.overlap_ratio*100}%)")
        print(f"Device: {self.device}")
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # List available devices
        print(f"\nAvailable audio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"  Device {i}: {info['name']} (In: {info['maxInputChannels']}, Out: {info['maxOutputChannels']})")
        
        # Setup input stream
        input_device_index = input_device if input_device is not None else p.get_default_input_device_info()['index']
        print(f"\nUsing input device: {p.get_device_info_by_index(input_device_index)['name']}")
        
        input_stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=self.chunk_size - self.overlap_size,
            stream_callback=self._audio_callback_input
        )
        
        # Setup output stream
        output_device_index = output_device if output_device is not None else p.get_default_output_device_info()['index']
        print(f"Using output device: {p.get_device_info_by_index(output_device_index)['name']}")
        
        output_stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=output_device_index,
            frames_per_buffer=self.chunk_size - self.overlap_size,
            stream_callback=self._audio_callback_output
        )
        
        # Setup visualization
        if visualize:
            self._setup_visualization()
        
        # Setup output recording
        recorded_audio = [] if save_output else None
        
        # Start processing thread
        self.running = True
        processing_thread = threading.Thread(target=self._processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start streams
        input_stream.start_stream()
        output_stream.start_stream()
        
        print(f"\nReal-time enhancement started!")
        print(f"Speak into your microphone. Press Ctrl+C to stop.")
        if visualize:
            print("Visualization window should appear.")
        
        try:
            last_stats_time = time.time()
            
            while True:
                time.sleep(0.1)
                
                # Print statistics every 5 seconds
                current_time = time.time()
                if current_time - last_stats_time > 5.0:
                    avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                    max_processing_time = np.max(self.processing_times) if self.processing_times else 0
                    
                    print(f"\nStats - Chunks processed: {self.total_chunks_processed}")
                    print(f"Avg processing time: {avg_processing_time*1000:.1f}ms")
                    print(f"Max processing time: {max_processing_time*1000:.1f}ms")
                    print(f"Input queue size: {self.input_queue.qsize()}")
                    print(f"Output queue size: {self.output_queue.qsize()}")
                    
                    last_stats_time = current_time
                
                # Record output if requested
                if save_output and not self.output_queue.empty():
                    try:
                        output_chunk = self.output_queue.queue[0]  # Peek without removing
                        recorded_audio.append(output_chunk.copy())
                    except:
                        pass
                
        except KeyboardInterrupt:
            print(f"\nStopping real-time enhancement...")
        
        finally:
            # Stop everything
            self.running = False
            input_stream.stop_stream()
            output_stream.stop_stream()
            input_stream.close()
            output_stream.close()
            p.terminate()
            
            # Save recorded output
            if save_output and recorded_audio:
                output_audio = np.concatenate(recorded_audio)
                sf.write(output_file, output_audio, self.sample_rate)
                print(f"Enhanced audio saved to: {output_file}")
            
            if visualize and self.fig:
                plt.close(self.fig)
            
            print("Real-time enhancement stopped.")


def main():
    parser = argparse.ArgumentParser(description='Real-time Voice Enhancement')
    parser.add_argument('--checkpoint', '-c', required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--reference', '-r', required=True,
                       help='Path to reference audio file (your clean voice)')
    parser.add_argument('--input-device', '-i', type=int, default=None,
                       help='Input audio device index (use list-devices to see options)')
    parser.add_argument('--output-device', '-o', type=int, default=None,
                       help='Output audio device index')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show real-time visualization')
    parser.add_argument('--save-output', '-s', action='store_true',
                       help='Save enhanced audio to file')
    parser.add_argument('--output-file', '-f', default='realtime_enhanced.wav',
                       help='Output filename for saved audio')
    parser.add_argument('--device', '-d', choices=['cpu', 'cuda'], default=None,
                       help='Processing device')
    parser.add_argument('--list-devices', '-l', action='store_true',
                       help='List available audio devices and exit')
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        import pyaudio
        p = pyaudio.PyAudio()
        print("Available audio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")
            print(f"  Max input channels: {info['maxInputChannels']}")
            print(f"  Max output channels: {info['maxOutputChannels']}")
            print(f"  Default sample rate: {info['defaultSampleRate']}")
            print()
        p.terminate()
        return
    
    # Validate files
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.reference):
        print(f"Error: Reference audio file not found: {args.reference}")
        return
    
    try:
        # Initialize enhancer
        enhancer = RealTimeVoiceEnhancer(
            checkpoint_path=args.checkpoint,
            reference_audio_path=args.reference,
            device=args.device
        )
        
        # Start real-time enhancement
        enhancer.start_enhancement(
            input_device=args.input_device,
            output_device=args.output_device,
            visualize=args.visualize,
            save_output=args.save_output,
            output_file=args.output_file
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()