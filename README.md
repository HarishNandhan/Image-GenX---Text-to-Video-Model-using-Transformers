


# **Image GenX - Text to Video Model using Transformers** 🚀



```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device_name = torch.cuda.get_device_name(0)  
print("Accelerator Device:", device_name)

model_name_or_path = "TheBloke/EverythingLM-13b-V2-16K-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                            device_map="auto",
                                            trust_remote_code=True,  
                                            revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "explain about law of attraction in 50 words"

print("\n\n*** Generate:")

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
generated_text = tokenizer.decode(output[0])
print(generated_text)

with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)
```

## Image Generation Module

```python
import os
import re
from gtts import gTTS
from moviepy.editor import *
from diffusers import StableDiffusionPipeline
import torch

# Read the generated text from a file
with open("generated_text.txt", "r") as file:
    text = file.read()

# Split text into paragraphs
paragraphs = re.split(r"[,.]", text)

# Create directories if they don't exist
os.makedirs("audio", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Initialize the StableDiffusionPipeline
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Get the name of the GPU device
device_name = torch.cuda.get_device_name(0)
print("Accelerator Device:", device_name)

# Initialize a list to store video clips
clips = []

# Iterate through paragraphs and create video clips
for i, para in enumerate(paragraphs, start=1):
    para = para.strip()

    if not para:
        continue

    # Create a prompt for the model
    prompt = f"photo,style,bright cinematic lighting, gopro, fisheye lens, {para}"

    # Generate an image using the model
    image = pipe(prompt).images[0]
    background_image_path = f"images/background{i}.png"
    image.save(background_image_path)

    # Convert paragraph to voiceover using gTTS
    tts = gTTS(text=para, lang='en', slow=False)
    tts.save(f"audio/voiceover{i}.mp3")
    print(f"The Paragraph {i} Converted into VoiceOver & Saved in Audio Folder!")

    # Load the audio file using moviepy
    audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
    audio_duration = audio_clip.duration

    # Create a text clip
    text_clip = TextClip(para, fontsize=50, color="white")
    text_clip = text_clip.set_pos('center').set_duration(audio_duration)

    # Create a background image clip
    background_clip = ImageClip(background_image_path, duration=audio_duration)

    # Combine the background image clip and text clip
    video_clip = CompositeVideoClip([background_clip.set_audio(audio_clip), text_clip])

    # Save the video clip to a file
    video_file_path = f"videos/video{i}.mp4"
    video_clip.write_videofile(video_file_path, fps=24)
    print(f"The Video {i} Has Been Created Successfully!")

    clips.append(video_clip)

# Concatenate all the video clips to create a final video
final_video = concatenate_videoclips(clips, method="compose")

if final_video is not None:
    # Resize the final video
    final_video = final_video.resize((1024, 1024))

    # Save the final video to a file
    final_video.write_videofile("final_video.mp4", fps=24)
    print("The Final Video Has Been Created Successfully!")
else:
    print("Error: Unable to create the final video.")
```

## Project Overview

The "Image GenX (Text To Video Model) Using Transformers" project combines advanced text and image generation techniques to create dynamic educational videos. The system utilizes the powerful GPT-3.5 language model to generate descriptive content based on user prompts.

### Text Generation

The text generation module generates concise and informative text based on user prompts. The generated text is saved to a file for later use.

### Image Generation

The image generation module employs the StableDiffusionPipeline, creating vivid background images based on the generated text. Each paragraph of text is associated with a unique background image.

### Video Creation

The generated text is converted into voiceovers, and the background images are combined with text to create individual video clips. These clips are then concatenated to produce a final educational video.

## Output Samples

### Generated Text

![Generated text](https://github.com/HarishNandhan/Image-GenX---Text-to-Video-Model-using-Transformers/blob/main/generated_text.jpg)

### Generated Image

![Generated Image](https://github.com/HarishNandhan/Image-GenX---Text-to-Video-Model-using-Transformers/blob/main/images/background32.png)

### Final Video

[![CLICK HERE](https://github.com/HarishNandhan/Image-GenX---Text-to-Video-Model-using-Transformers/blob/main/images/background50.png)](https://www.youtube.com/watch?v=i5JPjcB-ohA)
