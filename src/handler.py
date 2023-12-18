import runpod

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

from diffusers import DiffusionPipeline
import torch
import time 
import threading
import base64
from io import BytesIO


def load_base_pipeline():
    global base
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")


def load_refiner_pipeline():
    global refiner
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

def initialize_pipelines():
    base_thread = threading.Thread(target=load_base_pipeline)
    refiner_thread = threading.Thread(target=load_refiner_pipeline)

    base_thread.start()
    refiner_thread.start()

    base_thread.join()
    refiner_thread.join()

    return base, refiner


def generate_base_image(base, prompt, n_steps, high_noise_frac):
    return base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

def refine_image(refiner, prompt, n_steps, high_noise_frac, base_image, use_refiner):
    if use_refiner:
        return refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=base_image,
        ).images[0]
    else:
        return base_image[0]
    

def display_image(image, start_time):
    end_time = time.time() - start_time
    print(f"Time: {end_time:.2f} seconds")

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str




def handler(job):
    job_input = job['input']
    
    use_refiner = job_input.get('use_refiner', False)
    prompt = job_input.get('prompt')

    # Initialize pipelines
    base, refiner = initialize_pipelines()

    n_steps = 40
    high_noise_frac = 0.8

    start_time = time.time()

    base_image = generate_base_image(base, prompt, n_steps, high_noise_frac)
    final_image = refine_image(refiner, prompt, n_steps, high_noise_frac, base_image, use_refiner)

    image_base64 = display_image(final_image, start_time)

    name = job_input.get('name', 'World')
    return {"output": f"Hello, {name}! Image generated in {time.time() - start_time:.2f} seconds. Image Base64: {image_base64}"}


runpod.serverless.start({"handler": handler})
