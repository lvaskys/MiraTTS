import os
import torch
import soundfile as sf
import logging
import gradio as gr
import librosa
import numpy as np
from datetime import datetime
from mira.model import MiraTTS

# Configure logging for HF Spaces
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL = None

def initialize_model():
    """Initialize MiraTTS model with error handling for HF Spaces."""
    global MODEL
    
    if MODEL is not None:
        return MODEL
        
    try:
        logging.info("Initializing MiraTTS model...")
        model_dir = "YatharthS/MiraTTS"
        
        # Initialize with HF Spaces compatible settings
        MODEL = MiraTTS(
            model_dir=model_dir,
            tp=1,  # Single GPU
            enable_prefix_caching=False,  # Disable for stability
            cache_max_entry_count=0.1  # Reduced cache
        )
        
        logging.info("Model initialized successfully")
        return MODEL
        
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")
        raise e

def validate_audio_input(audio_path):
    """Validate and preprocess audio input for HF Spaces."""
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("Audio file not found")
    
    try:
        # Load and validate audio
        audio, sr = librosa.load(audio_path, sr=None, duration=30)  # Limit to 30s for memory
        
        if len(audio) == 0:
            raise ValueError("Audio file is empty")
        
        # Ensure minimum length
        min_length = int(0.5 * sr)  # At least 0.5 seconds
        if len(audio) < min_length:
            raise ValueError(f"Audio too short: {len(audio)/sr:.2f}s, minimum 0.5s required")
        
        # Resample to 16kHz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save preprocessed audio
        temp_dir = "/tmp" if os.path.exists("/tmp") else "."
        temp_path = os.path.join(temp_dir, f"processed_{os.path.basename(audio_path)}")
        sf.write(temp_path, audio, samplerate=sr)
        
        return temp_path, len(audio), sr
        
    except Exception as e:
        raise ValueError(f"Audio processing failed: {e}")

def generate_speech(text, prompt_audio_path):
    """Generate speech with GPU acceleration for HF Spaces."""
    try:
        # Initialize model if needed
        model = initialize_model()
        
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text input is empty")
            
        # Process audio
        processed_audio, length, sr = validate_audio_input(prompt_audio_path)
        logging.info(f"Audio processed: {length/sr:.2f}s at {sr}Hz")
        
        # Encode audio
        context_tokens = model.encode_audio(processed_audio)
        if context_tokens is None:
            raise ValueError("Failed to encode reference audio")
            
        # Generate speech
        output_audio = model.generate(text, context_tokens)
        if output_audio is None:
            raise ValueError("Speech generation failed")
        
        # Process output
        if torch.is_tensor(output_audio):
            output_audio = output_audio.cpu().numpy()
            
        if output_audio.dtype == 'float16':
            output_audio = output_audio.astype('float32')
        
        # Clean up
        if os.path.exists(processed_audio):
            os.remove(processed_audio)
            
        return output_audio, 48000
        
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise e

def voice_clone_interface(text, prompt_audio_upload, prompt_audio_record):
    """Interface for voice cloning."""
    try:
        if not text or not text.strip():
            return None, "Please enter text to synthesize."
            
        prompt_audio = prompt_audio_upload if prompt_audio_upload else prompt_audio_record
        if not prompt_audio:
            return None, "Please upload or record reference audio."
        
        # Generate audio
        audio, sample_rate = generate_speech(text, prompt_audio)
        
        # Save output
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/mira_tts_{timestamp}.wav"
        sf.write(output_path, audio, samplerate=sample_rate)
        
        return output_path, "Generation successful!"
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logging.error(error_msg)
        return None, error_msg

def build_interface():
    """Build Gradio interface optimized for HF Spaces."""
    
    with gr.Blocks(title="MiraTTS - Voice Cloning") as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1 style="color: #2563eb; margin-bottom: 10px;">MiraTTS Voice Cloning</h1>
            <p style="color: #64748b; font-size: 16px;">
                High-quality voice synthesis with 100x realtime speed using optimized LMDeploy
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Reference Audio")
                
                prompt_upload = gr.Audio(
                    sources="upload",
                    type="filepath",
                    label="Upload Reference Audio"
                )
                
                gr.Markdown("*Upload a clear audio sample (3-30 seconds, 16kHz+)*")
                
                prompt_record = gr.Audio(
                    sources="microphone",
                    type="filepath", 
                    label="Record Reference Audio"
                )
                
                gr.Markdown("*Record directly in your browser*")
                
            with gr.Column(scale=1):
                gr.Markdown("### Text Input")
                
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=4,
                    value="Hello! This is a demonstration of MiraTTS, an optimized text-to-speech model."
                )
                
                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary"
                )
        
        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Speech",
                    type="filepath",
                    autoplay=True
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=True
                )
        
        # Examples
        gr.Markdown("### Example Usage")
        gr.Markdown("""
        1. **Upload or record** a reference audio (your target voice)
        2. **Enter text** you want to synthesize in that voice
        3. **Click generate** and wait for the result
        
        **Tips:**
        - Use clear reference audio without background noise
        - Keep reference audio between 3-30 seconds
        - Shorter text generates faster
        """)
        
        # Event handlers
        generate_btn.click(
            voice_clone_interface,
            inputs=[text_input, prompt_upload, prompt_record],
            outputs=[output_audio, status_text],
            show_progress=True
        )
        
        # Clear function
        def clear_all():
            return None, None, "", None, "Ready for new generation"
            
        clear_btn = gr.Button("Clear All", variant="secondary")
        clear_btn.click(
            clear_all,
            outputs=[prompt_upload, prompt_record, text_input, output_audio, status_text]
        )
    
    return demo

if __name__ == "__main__":

    # Removing this for zero-gpu to work.
    # Initialize model at startup
    #try:
    #    initialize_model()
    #    logging.info("Model pre-loaded successfully")
    #except Exception as e:
    #    logging.error(f"Failed to pre-load model: {e}")
    
    # Launch interface
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )