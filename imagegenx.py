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
