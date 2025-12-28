import gc
import torch
from itertools import cycle
from ncodec.codec import TTSCodec
from vllm import LLM, SamplingParams

from mira.utils import clear_cache, split_text

class MiraTTS:

    def __init__(self, model_dir="YatharthS/MiraTTS", tp=1, enable_prefix_caching=True, cache_max_entry_count=0.6):
        
        # Initialize vLLM
        
        self.pipe = LLM(
            model=model_dir,
            tensor_parallel_size=tp,
            dtype='bfloat16',
            gpu_memory_utilization=cache_max_entry_count,
            enable_prefix_caching=enable_prefix_caching,
            quantization=None,
            trust_remote_code=True
        )

        self.gen_config = SamplingParams(
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            repetition_penalty=1.2,
            min_p=0.05,
            max_tokens=1024
        )
        self.codec = TTSCodec()

    def set_params(self, top_p=0.95, top_k=50, temperature=0.8, max_new_tokens=1024, repetition_penalty=1.2, min_p=0.05):
        """sets sampling parameters for the llm"""
      
        self.gen_config = SamplingParams(
            top_p=top_p, 
            top_k=top_k, 
            temperature=temperature, 
            max_tokens=max_new_tokens, 
            repetition_penalty=repetition_penalty, 
            min_p=min_p
        )
      
    def c_cache(self):
        clear_cache()

    def split_text(self, text):
        return split_text(text)
        
    def encode_audio(self, audio_file):
        """encodes audio into context tokens"""
      
        context_tokens = self.codec.encode(audio_file)
        return context_tokens

        
    def generate(self, text, context_tokens):
        """generates speech from input text"""
        formatted_prompt = self.codec.format_prompt(text, context_tokens, None)
      
        # vLLM generate takes a list of prompts or single prompt
        # Returns list of RequestOutput
        outputs = self.pipe.generate([formatted_prompt], sampling_params=self.gen_config, use_tqdm=False)
        response_text = outputs[0].outputs[0].text
        audio = self.codec.decode(response_text, context_tokens)
        return audio
      
    def batch_generate(self, prompts, context_tokens):
        """
        Generates speech from text, for larger batch size

        Args:
            prompt (list): Input for tts model, list of prompts
            voice (list): Description of voice, list of voices respective to prompt
        """
        formatted_prompts = []
        for prompt, context_token in zip(prompts, cycle(context_tokens)):
            formatted_prompt = self.codec.format_prompt(prompt, context_token, None)
            formatted_prompts.append(formatted_prompt)
        
        outputs = self.pipe.generate(formatted_prompts, sampling_params=self.gen_config, use_tqdm=True)
        generated_tokens = [output.outputs[0].text for output in outputs]
      
        audios = []
        for generated_token, context_token in zip(generated_tokens, cycle(context_tokens)):
            audio = self.codec.decode(generated_token, context_token)
            audios.append(audio)
        audios = torch.cat(audios, dim=0)
      
        return audios
            

