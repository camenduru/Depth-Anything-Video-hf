import gradio as gr
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import tempfile
import spaces 

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

@torch.no_grad()
def predict_depth(model, image):
    return model(image)["depth"]

@spaces.GPU
def make_video(video_path, outdir='./vis_video_depth',encoder='vitl'):
    if encoder not in ["vitl","vitb","vits"]:
        encoder = "vits"
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(DEVICE).eval()
    # Define path for temporary processed frames
    temp_frame_dir = tempfile.mkdtemp()
    
    margin_width = 50

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = "cuda"
    # depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    depth_anything = pipeline(task = "depth-estimation", model="nielsr/depth-anything-small", device=0)
    
    # total_params = sum(param.numel() for param in depth_anything.parameters())
    # print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(video_path):
        if video_path.endswith('txt'):
            with open(video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [video_path]
    else:
        filenames = os.listdir(video_path)
        filenames = [os.path.join(video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    # os.makedirs(outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        
        filename = os.path.basename(filename)
        # output_path = os.path.join(outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            output_path = tmpfile.name
        #out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (output_width, frame_height))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (output_width, frame_height))
        # count=0
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            frame_pil =  Image.fromarray((frame * 255).astype(np.uint8))
            frame = transform({'image': frame})['image']
            
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
            
            
            depth = predict_depth(depth_anything, frame_pil)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
            
            # out.write(combined_frame)
            # frame_path = os.path.join(temp_frame_dir, f"frame_{count:05d}.png")
            # cv2.imwrite(frame_path, combined_frame)
            out.write(combined_frame)
            # count += 1
        
        raw_video.release()
        out.release()
        return output_path

css = """
#img-display-container {
    max-height: 100vh;
    }
#img-display-input {
    max-height: 80vh;
    }
#img-display-output {
    max-height: 80vh;
    }
"""


title = "# Depth Anything Video Demo"
description = """Depth Anything on full video files.

Please refer to our [paper](https://arxiv.org/abs/2401.10891), [project page](https://depth-anything.github.io), or [github](https://github.com/LiheYoung/Depth-Anything) for more details."""

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])

# @torch.no_grad()
# def predict_depth(model, image):
#     return model(image)

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Video Depth Prediction demo")

    with gr.Row():
        input_video = gr.Video(label="Input Video")
        model_type = gr.Dropdown(["vits", "vitb", "vitl"], type="value", label='Model Type')
    submit = gr.Button("Submit")
    processed_video = gr.Video(label="Processed Video")

    def on_submit(uploaded_video,model_type):
                
        # Process the video and get the path of the output video
        output_video_path = make_video(uploaded_video,encoder=model_type)

        return output_video_path

    submit.click(on_submit, inputs=[input_video, model_type], outputs=processed_video)

    example_files = os.listdir('assets/examples_video')
    example_files.sort()
    example_files = [os.path.join('assets/examples_video', filename) for filename in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_video], outputs=processed_video, fn=on_submit, cache_examples=True)
    

if __name__ == '__main__':
    demo.queue().launch()