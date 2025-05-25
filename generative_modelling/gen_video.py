# import torch
# from diffusers import LTXPipeline
# from diffusers.utils import export_to_video

# pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to("cuda")

import os
import time

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    # device_map="cuda",
).to("cuda")


prompt = "Video of the inside of a home, quickly moving through the different rooms, clear, fast, dynamic"
# negative_prompt = "worst quality, motion, blurry, jittery, distorted, dynamic"
negative_prompt = ""

video = pipe(
    prompt=prompt,
    # negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_frames=168,
    num_inference_steps=50,
).frames[0]

path_to_save = "/scratch/toskov/geometry_consistency/gen_videos/"
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
n_existing_vids = len(os.listdir(path_to_save))

# Get current timestamp to avoid overwriting
current_time = time.time_ns()

vid_path = os.path.join(
    path_to_save,
    f"video_{n_existing_vids + 1}_time_{current_time}_hunyuan_'{prompt.replace(' ', '-')}'_.mp4",
)


export_to_video(video, vid_path, fps=24)
