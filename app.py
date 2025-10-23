import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

pipeline_image = None
pipeline_text = None

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    return gs, mesh

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def preprocess_image(image: Image.Image) -> Image.Image:
    global pipeline_image
    if pipeline_image is None:
        pipeline_image = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline_image.cuda()
    processed_image = pipeline_image.preprocess_image(image)
    return processed_image

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    global pipeline_image
    if pipeline_image is None:
        pipeline_image = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline_image.cuda()
    imgs = [img[0] for img in images]
    return [pipeline_image.preprocess_image(img) for img in imgs]

def split_image(image: Image.Image) -> List[Image.Image]:
    image_np = np.array(image)
    alpha = image_np[..., 3]
    alpha_mask = np.any(alpha > 0, axis=0)
    start_pos = np.where(~alpha_mask[:-1] & alpha_mask[1:])[0].tolist()
    end_pos = np.where(alpha_mask[:-1] & ~alpha_mask[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image_np[:, s:e+1]))
    return [preprocess_image(img) for img in images]

def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images

def switch_to_image():
    global pipeline_image, pipeline_text
    if pipeline_text is not None:
        del pipeline_text
        torch.cuda.empty_cache()
        pipeline_text = None
    if pipeline_image is None:
        pipeline_image = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline_image.cuda()
    return gr.State().update(value=None)

def switch_to_text():
    global pipeline_image, pipeline_text
    if pipeline_image is not None:
        del pipeline_image
        torch.cuda.empty_cache()
        pipeline_image = None
    if pipeline_text is None:
        pipeline_text = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        pipeline_text.cuda()
    return gr.State().update(value=None)

def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: str,
    req: gr.Request,
) -> Tuple[dict, str]:
    global pipeline_image
    if pipeline_image is None:
        pipeline_image = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline_image.cuda()
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if not is_multiimage:
        outputs = pipeline_image.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        imgs = [img[0] for img in multiimages]
        outputs = pipeline_image.run_multi_image(
            imgs,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path

def text_to_3d(
    prompt: str,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    req: gr.Request,
) -> Tuple[dict, str]:
    global pipeline_text
    if pipeline_text is None:
        pipeline_text = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        pipeline_text.cuda()
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    outputs = pipeline_text.run(
        prompt,
        seed=seed,
        formats=["gaussian", "mesh"],
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path

def extract_glb(state: dict, mesh_simplify: float, texture_size: int, req: gr.Request) -> Tuple[str, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path

def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path

with gr.Blocks() as demo:
    with gr.Tabs() as main_tabs:
        with gr.Tab(label="Image to 3D", id=0) as image_tab:
            gr.Markdown("""
## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
* Upload an image and click "Generate" to create a 3D asset. If the image has an alpha channel, it will be used as the mask. Otherwise, we use `rembg` to remove the background.
* If you find the generated 3D asset satisfactory, click "Extract GLB" to extract the GLB file and download it.
""")
            with gr.Row():
                with gr.Column():
                    with gr.Tabs() as input_tabs:
                        with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=300)
                        with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                            multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=300, columns=3)
                            gr.Markdown("""
Input different views of the object in separate images.

*NOTE: this is an experimental algorithm without training a specialized model. It may not produce the best results for all images, especially those having different poses or inconsistent details.*
""")
                    with gr.Accordion(label="Generation Settings", open=False):
                        seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                        randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                        gr.Markdown("Stage 2: Structured Latent Generation")
                        with gr.Row():
                            slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                            slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                        multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="stochastic")
                    generate_btn = gr.Button("Generate")
                    with gr.Accordion(label="GLB Extraction Settings", open=False):
                        mesh_simplify = gr.Slider(0.0, 0.98, label="Simplify", value=0.95, step=0.01)
                        texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
                    with gr.Row():
                        extract_glb_btn = gr.Button("Extract GLB", interactive=True)
                        extract_gs_btn = gr.Button("Extract Gaussian", interactive=True)
                    gr.Markdown("""
*NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
""")
                with gr.Column():
                    image_video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
                    image_model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)
                    with gr.Row():
                        image_download_glb = gr.DownloadButton(label="Download GLB", interactive=True)
                        image_download_gs = gr.DownloadButton(label="Download Gaussian", interactive=True)
            is_multiimage = gr.State(False)
            image_output_buf = gr.State()
            with gr.Row() as single_image_example:
                examples = gr.Examples(
                    examples=[f'assets/example_image/{image}' for image in os.listdir("assets/example_image")],
                    inputs=[image_prompt],
                    fn=preprocess_image,
                    outputs=[image_prompt],
                    run_on_click=True,
                    examples_per_page=64,
                )
            with gr.Row(visible=False) as multiimage_example:
                examples_multi = gr.Examples(
                    examples=prepare_multi_example(),
                    inputs=[image_prompt],
                    fn=split_image,
                    outputs=[multiimage_prompt],
                    run_on_click=True,
                    examples_per_page=8,
                )
            single_image_input_tab.select(
                lambda: (False, gr.Row.update(visible=True), gr.Row.update(visible=False)),
                outputs=[is_multiimage, single_image_example, multiimage_example]
            )
            multiimage_input_tab.select(
                lambda: (True, gr.Row.update(visible=False), gr.Row.update(visible=True)),
                outputs=[is_multiimage, single_image_example, multiimage_example]
            )
            image_prompt.upload(
                preprocess_image,
                inputs=[image_prompt],
                outputs=[image_prompt],
            )
            multiimage_prompt.upload(
                preprocess_images,
                inputs=[multiimage_prompt],
                outputs=[multiimage_prompt],
            )
            generate_btn.click(
                get_seed,
                inputs=[randomize_seed, seed],
                outputs=[seed],
            ).then(
                image_to_3d,
                inputs=[image_prompt, multiimage_prompt, is_multiimage, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo],
                outputs=[image_output_buf, image_video_output],
            ).then(
                lambda: (gr.Button.update(interactive=True), gr.Button.update(interactive=True)),
                outputs=[extract_glb_btn, extract_gs_btn],
            )
            image_video_output.clear(
                lambda: (gr.Button.update(interactive=False), gr.Button.update(interactive=False)),
                outputs=[extract_glb_btn, extract_gs_btn],
            )
            extract_glb_btn.click(
                extract_glb,
                inputs=[image_output_buf, mesh_simplify, texture_size],
                outputs=[image_model_output, image_download_glb],
            ).then(
                lambda: gr.Button.update(interactive=True),
                outputs=[image_download_glb],
            )
            extract_gs_btn.click(
                extract_gaussian,
                inputs=[image_output_buf],
                outputs=[image_model_output, image_download_gs],
            ).then(
                lambda: gr.Button.update(interactive=True),
                outputs=[image_download_gs],
            )
            image_model_output.clear(
                lambda: gr.Button.update(interactive=False),
                outputs=[image_download_glb],
            )

        with gr.Tab(label="Text to 3D", id=1) as text_tab:
            gr.Markdown("""
## Text to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
* Type a text prompt and click "Generate" to create a 3D asset.
* If you find the generated 3D asset satisfactory, click "Extract GLB" to extract the GLB file and download it.
""")
            with gr.Row():
                with gr.Column():
                    text_prompt = gr.Textbox(label="Text Prompt", lines=5)
                    with gr.Accordion(label="Generation Settings", open=False):
                        text_seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                        text_randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            text_ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            text_ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)
                        gr.Markdown("Stage 2: Structured Latent Generation")
                        with gr.Row():
                            text_slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            text_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)
                    text_generate_btn = gr.Button("Generate")
                    with gr.Accordion(label="GLB Extraction Settings", open=False):
                        text_mesh_simplify = gr.Slider(0.0, 0.98, label="Simplify", value=0.95, step=0.01)
                        text_texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
                    with gr.Row():
                        text_extract_glb_btn = gr.Button("Extract GLB", interactive=True)
                        text_extract_gs_btn = gr.Button("Extract Gaussian", interactive=True)
                    gr.Markdown("""
*NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
""")
                with gr.Column():
                    text_video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
                    text_model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)
                    with gr.Row():
                        text_download_glb = gr.DownloadButton(label="Download GLB", interactive=True)
                        text_download_gs = gr.DownloadButton(label="Download Gaussian", interactive=True)
            text_output_buf = gr.State()
            text_generate_btn.click(
                get_seed,
                inputs=[text_randomize_seed, text_seed],
                outputs=[text_seed],
            ).then(
                text_to_3d,
                inputs=[text_prompt, text_seed, text_ss_guidance_strength, text_ss_sampling_steps, text_slat_guidance_strength, text_slat_sampling_steps],
                outputs=[text_output_buf, text_video_output],
            ).then(
                lambda: (gr.Button.update(interactive=True), gr.Button.update(interactive=True)),
                outputs=[text_extract_glb_btn, text_extract_gs_btn],
            )
            text_video_output.clear(
                lambda: (gr.Button.update(interactive=False), gr.Button.update(interactive=False)),
                outputs=[text_extract_glb_btn, text_extract_gs_btn],
            )
            text_extract_glb_btn.click(
                extract_glb,
                inputs=[text_output_buf, text_mesh_simplify, text_texture_size],
                outputs=[text_model_output, text_download_glb],
            ).then(
                lambda: gr.Button.update(interactive=True),
                outputs=[text_download_glb],
            )
            text_extract_gs_btn.click(
                extract_gaussian,
                inputs=[text_output_buf],
                outputs=[text_model_output, text_download_gs],
            ).then(
                lambda: gr.Button.update(interactive=True),
                outputs=[text_download_gs],
            )
            text_model_output.clear(
                lambda: gr.Button.update(interactive=False),
                outputs=[text_download_glb],
            )

    state_dummy = gr.State()
    image_tab.select(switch_to_image, outputs=[state_dummy])
    text_tab.select(switch_to_text, outputs=[state_dummy])

    demo.load(start_session)
    demo.unload(end_session)

if __name__ == "__main__":
    demo.launch()
