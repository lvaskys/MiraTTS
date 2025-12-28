import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
from datetime import datetime
from mira.model import MiraTTS

MODEL = None

def initialize_model(model_dir="YatharthS/MiraTTS"):
    """Load the MiraTTS model once at the beginning."""
    logging.info(f"Loading MiraTTS model from: {model_dir}")
    model = MiraTTS(model_dir)
    return model

def generate_audio(text, prompt_audio_path):
    """Generate audio from text using MiraTTS with voice cloning."""
    global MODEL
    
    if MODEL is None:
        MODEL = initialize_model()
    
    try:
        # Encode the prompt audio
        context_tokens = MODEL.encode_audio(prompt_audio_path)
        
        # Generate audio
        audio = MODEL.generate(text, context_tokens)
        
        # Convert to numpy array if it's a tensor and handle dtype
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        # Ensure correct dtype for soundfile (convert from float16 to float32)
        if audio.dtype == 'float16':
            audio = audio.astype('float32')
        elif audio.dtype not in ['float32', 'float64', 'int16', 'int32']:
            audio = audio.astype('float32')
            
        return audio, 48000  # Return audio and sample rate
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        raise e

def run_tts(text, prompt_audio_path, save_dir="results"):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"mira_tts_{timestamp}.wav")
    
    logging.info("Starting MiraTTS inference...")
    
    # Generate audio
    audio, sample_rate = generate_audio(text, prompt_audio_path)
    
    # Save audio file
    sf.write(save_path, audio, samplerate=sample_rate)
    
    logging.info(f"Audio saved at: {save_path}")
    return save_path

def voice_clone_callback(text, prompt_audio_upload, prompt_audio_record):
    """Gradio callback for voice cloning using MiraTTS."""
    if not text.strip():
        return None
        
    # Use uploaded audio or recorded audio
    prompt_audio = prompt_audio_upload if prompt_audio_upload else prompt_audio_record
    
    if not prompt_audio:
        return None
        
    try:
        audio_output_path = run_tts(text, prompt_audio)
        return audio_output_path
    except Exception as e:
        logging.error(f"Error in voice cloning: {e}")
        return None

def voice_creation_callback(text, temperature, top_p, top_k):
    """Gradio callback for creating synthetic voice with custom parameters."""
    if not text.strip():
        return None
        
    global MODEL
    
    if MODEL is None:
        MODEL = initialize_model()
    
    try:
        # Set custom generation parameters
        MODEL.set_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=1024,
            repetition_penalty=1.2
        )
        
        # Use a default voice context (you may want to provide default audio files)
        # Check multiple possible paths for example audio
        possible_paths = [
            "/models3/src/MiraTTS/models/MiraTTS/example1.wav",
            "models/MiraTTS/example1.wav",
            "./models/MiraTTS/example1.wav"
        ]
        
        default_audio = None
        for path in possible_paths:
            if os.path.exists(path):
                default_audio = path
                break
        
        if default_audio:
            # Generate audio with dtype conversion
            context_tokens = MODEL.encode_audio(default_audio)
            audio = MODEL.generate(text, context_tokens)
            
            # Handle tensor conversion and dtype
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()
            
            # Ensure correct dtype for soundfile
            if audio.dtype == 'float16':
                audio = audio.astype('float32')
            elif audio.dtype not in ['float32', 'float64', 'int16', 'int32']:
                audio = audio.astype('float32')
            
            # Save the audio
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join("results", f"mira_tts_creation_{timestamp}.wav")
            sf.write(save_path, audio, samplerate=48000)
            
            return save_path
        else:
            logging.warning("No default audio found for voice creation")
            return None
            
    except Exception as e:
        logging.error(f"Error in voice creation: {e}")
        return None

def build_ui():
    """Build the Gradio interface similar to SparkTTS."""
    
    with gr.Blocks(title="MiraTTS Web Interface") as demo:
        # Title
        gr.HTML('<h1 style="text-align: center;">MiraTTS - High Quality Voice Synthesis</h1>')
        
        # Description
        gr.Markdown("""
        MiraTTS is a highly optimized Text-to-Speech model based on Spark-TTS with LMDeploy acceleration.
        It provides over 100x realtime generation speed with high-quality 48kHz audio output.
        """)
        
        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown("### Clone any voice using a reference audio sample")
                
                with gr.Row():
                    prompt_audio_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        label="Upload Reference Audio (recommended: 3-30 seconds, 16kHz+)",
                    )
                    prompt_audio_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Record Reference Audio",
                    )
                
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    lines=3,
                    placeholder="Enter the text you want to convert to speech...",
                    value="Hello! This is a demonstration of MiraTTS voice cloning capabilities."
                )
                
                with gr.Row():
                    clone_button = gr.Button("Generate Audio", variant="primary")
                    clear_button = gr.Button("Clear")
                
                audio_output_clone = gr.Audio(
                    label="Generated Audio",
                    autoplay=True
                )
                
                clone_button.click(
                    voice_clone_callback,
                    inputs=[text_input, prompt_audio_upload, prompt_audio_record],
                    outputs=[audio_output_clone],
                )
                
                clear_button.click(
                    lambda: (None, None, "", None),
                    outputs=[prompt_audio_upload, prompt_audio_record, text_input, audio_output_clone]
                )
            
            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.Markdown("### Create synthetic voices with custom parameters")
                
                with gr.Row():
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Text to Synthesize",
                            lines=3,
                            placeholder="Enter text here...",
                            value="You can create customized voices by adjusting the generation parameters below."
                        )
                        
                        with gr.Row():
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.5,
                                step=0.1,
                                value=0.8,
                                label="Temperature (creativity)"
                            )
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.05,
                                value=0.95,
                                label="Top-p (nucleus sampling)"
                            )
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Top-k (vocabulary size)"
                            )
                    
                    with gr.Column():
                        create_button = gr.Button("Create Voice", variant="primary")
                        audio_output_creation = gr.Audio(
                            label="Generated Audio",
                            autoplay=True
                        )
                
                create_button.click(
                    voice_creation_callback,
                    inputs=[text_input_creation, temperature, top_p, top_k],
                    outputs=[audio_output_creation],
                )
            
            # About Tab
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About MiraTTS
                
                MiraTTS is an optimized version of Spark-TTS with the following features:
                
                - **Ultra-fast generation**: Over 100x realtime speed using LMDeploy optimization
                - **High quality**: Generates crisp 48kHz audio outputs
                - **Memory efficient**: Works within 6GB VRAM
                - **Low latency**: As low as 100ms generation time
                - **Voice cloning**: Clone any voice from a short audio sample
                
                ### Model Information
                - Base model: Spark-TTS-0.5B
                - Optimization: LMDeploy + FlashSR
                - Sample rate: 48kHz
                - Model size: ~500M parameters
                
                ### Usage Tips
                - For voice cloning, use clear audio samples between 3-30 seconds
                - Ensure reference audio is at least 16kHz quality
                - Longer text inputs may require more memory
                - Adjust generation parameters for different voice styles
                """)
    
    return demo

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MiraTTS Gradio Web Interface")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="YatharthS/MiraTTS",
        help="Path to the MiraTTS model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="127.0.0.1",
        help="Server host/IP for Gradio app"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize model
    logging.info("Initializing MiraTTS model...")
    MODEL = initialize_model(args.model_dir)
    
    # Build and launch interface
    logging.info("Building Gradio interface...")
    demo = build_ui()
    
    logging.info(f"Launching web interface on {args.server_name}:{args.server_port}")
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )