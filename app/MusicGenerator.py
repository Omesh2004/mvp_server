from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import os
import torch

# Create output directory if it doesn't exist
os.makedirs("out", exist_ok=True)

# Load model and processor (this will happen once when the file is imported)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", device_map="auto")
model.eval()

def generate_music(prompt, duration):
    """
    Generate music based on a text prompt
    
    Args:
        prompt (str): Text description of the music to generate
        duration (float): Duration in seconds (approximate)
        
    Returns:
        str: Path to the generated audio file
    """
    # Define output path
    output_path = os.path.join("out", "generated.mp3")
    
    # Delete old file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Process the input
    inputs = processor(
        text=prompt,
        padding=True,
        return_tensors="pt",
    )
    
    # Generate audio values (adjust tokens based on duration)
    audio_values = model.generate(
        **inputs,
        max_new_tokens=int(256*(duration/5))
    )
    
    # Get sampling rate from model config
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    # Save the audio file
    scipy.io.wavfile.write(
        output_path,
        rate=sampling_rate,
        data=audio_values[0, 0].cpu().numpy()
    )
    
    return output_path