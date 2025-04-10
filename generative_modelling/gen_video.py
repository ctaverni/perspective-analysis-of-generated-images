# import torch
# from diffusers import LTXPipeline
# from diffusers.utils import export_to_video

# pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to("cuda")

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
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


prompt = "Video of the inside of a home, quickly moving through the different rooms, clear, fast, dynamic, unblurred, smooth"
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
export_to_video(video, f'output_hunyuan_"{prompt}".mp4', fps=24)