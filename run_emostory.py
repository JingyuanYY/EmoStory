import argparse
import os
from typing import List, Tuple, Dict
import torch
from emostory_pipeline import EmoStoryPipeline
from attention_store import AttentionStore
from emostory_transformer import EmoStoryTransformer2DModel
import time
from transformers import T5EncoderModel, CLIPTextModel
from utils.utils_function import find_token_indices_via_offsets_batch, get_element_tokens_mask, debug_recover_all
import json
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mutual storyboards with FluxPipeline")
    parser.add_argument("--emotion", type=str, default="amusement", help="emotion")
    parser.add_argument("--worker_id", type=int, default=0, help="device")
    parser.add_argument("--worker_num", type=int, default=4, help="device num")
    parser.add_argument("--boost_factor", type=float, default=1.5, help="Boost factor for attention.")
    parser.add_argument("--agent_result_path", type=str, default="agent_result.json", help="agent result path")

    parser.add_argument('--is_enhance_elements', action="store_true", help='is enhance elements in attention.')
    parser.add_argument('--save_cross_attn_weights', action="store_true", help='Save cross attention weights.')
    parser.add_argument("--save_step", type=List[int], default=[5, 10, 15, 20, 25], help="save step")
    parser.add_argument('--subject', type=str, default="child", help='Main subject of the storyboard.')
    parser.add_argument("--elements", type=str, default=[])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--guidance', type=float, default=3.5,
                        help='Classifier-free guidance scale (higher values = closer `to prompt, lower = more diverse)')
    parser.add_argument('--same_noise', type=bool, default=True,
                        help='Use the same initial noise tensor for all images in the batch')
    parser.add_argument('--ravm_mixing_coef', type=float, default=0.5,
                        help='Region-aware Story Generation')
    parser.add_argument('--first_mixing_block', type=int, default=30,
                        help='First transformer block index where Region-aware is applied')
    parser.add_argument('--last_mixing_block', type=int, default=57,
                        help='Last transformer block index where Region-aware is applied')
    parser.add_argument('--first_mixing_denoising_step', type=int, default=1,
                        help='First denoising step where Region-aware is applied')
    parser.add_argument('--last_mixing_denoising_step', type=int, default=21,
                        help='Last denoising step where Region-aware is applied')
    parser.add_argument('--n_diff_steps', type=int, default=28,
                        help='Total number of denoising steps')

    parser.add_argument(
        '--ref_image_prompt', type=str, default="child",
        help='Prompt describing the reference (top) panel.')
    parser.add_argument(
        '--story_prompts', nargs='+', type=str, default=[],
        help='List of panel prompts (space-separated).')
    parser.add_argument(
        '--output_dir', type=str, default="/mnt/d/dataset/Story2Board",
        help='Directory where generated images will be saved.')

    args = parser.parse_args()

    # Validate arguments
    if args.first_mixing_denoising_step < 1:
        parser.error("--first_mixing_denoising_step must be at least 1.")
    # Construct full prompts
    args.prompts = [
        f"A story of {args.ref_image_prompt} (top) and "
        f"the {story_prompt} (bottom)"
        for story_prompt in args.story_prompts
    ]
    # entire body visible, wide shot, long shot
    return args


args = parse_args()


def save_story_images(images, output_dir, json_output_dir, prompts, story_data):
    #current_time_str = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())  # 随时取新时间
    def split_image(image):
        width, height = image.size
        top_half = image.crop((0, 0, width, height // 2))
        bottom_half = image.crop((0, height // 2, width, height))
        return top_half, bottom_half

    panel_index = 0
    all_image_paths = []

    for i, image in enumerate(images):
        top, bottom = split_image(image)

        if i == 0:
            panel = top
            filename = f"ref_image.png"
            path = os.path.join(output_dir, filename)
            panel.save(path)
            print(f"Saved ref panel at: {path}")

        panel = bottom
        filename = f"image_{panel_index}.png"
        path = os.path.join(output_dir, filename)
        panel.save(path)
        all_image_paths.append(path)
        print(f"Saved panel at: {path}")
        panel_index += 1

    story_data["seed"] = args.seed
    story_data["actual_prompts"] = prompts
    story_data["image_paths"] = all_image_paths

    with open(json_output_dir, "w", encoding="utf-8") as f:
        json.dump(story_data, f, ensure_ascii=False, indent=4)

    print("Seeds, prompts, and image paths have been saved")


def run_mutual_story(pipe, prompts, seed, guidance, n_diff_steps, same_noise, ravm_mixing_coef,
    first_mixing_block, last_mixing_block, first_mixing_denoising_step, last_mixing_denoising_step,
    is_enhance_elements, boost_factor, dtype, device=0, output_dir="./", args=None
):
    all_token_indices = {}
    # target = {
    #     "subject": { "dog" : [[1,2,3], [6,4], [2,3,4],[3,4]] },
    #     "elements": { "flower" : [[1,2,3], None, [2,3,4],[3,4]], "cat" ... },
    # }
    tokenizer = pipe.tokenizer_2
    plist = find_token_indices_via_offsets_batch(args.subject, prompts, tokenizer)
    all_token_indices["subject"] = {args.subject: plist}
    all_token_indices["elements"] = {}
    for element in args.elements:
        plist = find_token_indices_via_offsets_batch(element, prompts, tokenizer)
        all_token_indices["elements"][element] = plist

    print("***************************************")
    print("all_token_indices: ", all_token_indices)

    debug_recover_all(all_token_indices, prompts, tokenizer)

    # Generate the token mask in advance
    p_mask = torch.zeros((len(prompts), 4608), dtype=dtype, device=device)
    p_mask = get_element_tokens_mask(p_mask, all_token_indices)

    EmoStory_transformer = pipe.transformer
    EmoStory_transformer.reset_attn_processor(
        is_enhance_elements=is_enhance_elements,
        save_cross_attn_weights=args.save_cross_attn_weights,
        emotion_token_indices=all_token_indices,
        p_mask=p_mask,
        boost_factor=boost_factor,
        save_step=args.save_step
    )

    n_transformer_blocks = EmoStory_transformer.num_layers + EmoStory_transformer.num_single_layers
    n_prompt_tokens = pipe.tokenizer_2.model_max_length
    n_attn_heads = EmoStory_transformer.num_attention_heads
    n_image_tokens = int(EmoStory_transformer.joint_attention_dim)
    print(f"n_image_tokens: {n_image_tokens}, n_prompt_tokens: {n_prompt_tokens}", )
    # == 4096, 512

    print("story prompts: ")
    print(prompts)

    attention_store = AttentionStore(
        batch_size=len(prompts),
        n_diff_steps=n_diff_steps,
        n_trans_blocks=n_transformer_blocks,
        n_prompt_tokens=n_prompt_tokens,
        n_image_tokens=n_image_tokens,
        n_attn_heads=n_attn_heads,
        dtype=dtype,
        device=device,
        output_dir=output_dir,
        emotion_token_indices=all_token_indices
    )

    # Arguments to be passed to the attention processor
    attention_args = {
        'attn_store': attention_store,
        'n_prompt_tokens': n_prompt_tokens,
        'n_image_tokens': n_image_tokens,
        'ravm_mixing_coef': ravm_mixing_coef,
        'first_mixing_block': first_mixing_block,
        'last_mixing_block': last_mixing_block,
        'first_mixing_denoising_step': first_mixing_denoising_step,
        'last_mixing_denoising_step': last_mixing_denoising_step,
    }
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    images = pipe(
        prompts,
        height=1024,
        width=1024,
        guidance_scale=guidance,
        num_inference_steps=n_diff_steps,
        generator=generator,
        same_noise=same_noise,
        output_dir=output_dir,
        save_step=args.save_step,
        joint_attention_kwargs=attention_args,
    ).images

    return images, attention_store


def load_model():
    first_mixing_block = args.first_mixing_block
    last_mixing_block = args.last_mixing_block
    first_mixing_denoising_step = args.first_mixing_denoising_step
    last_mixing_denoising_step = args.last_mixing_denoising_step
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    device_1 = "cuda:0"
    device_2 = "cuda:1"
    flux_id = "/mnt/d/crc/models/FLUX.1-dev"

    text_encoder = CLIPTextModel.from_pretrained(
        flux_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        # device_map="balanced"
    ).to(device_2)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        flux_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16,
        # device_map="balanced"
    ).to(device_2)

    es_transformer = EmoStoryTransformer2DModel.from_pretrained(
        flux_id,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="balanced",
    )

    # Create custom Story2Board pipeline
    pipe = EmoStoryPipeline.from_pretrained(
        flux_id,
        torch_dtype=dtype,
        # Custom emostory transformer
        transformer=es_transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        # RAVM hyperparameters
        first_mixing_block=first_mixing_block,
        last_mixing_block=last_mixing_block,
        first_mixing_denoising_step=first_mixing_denoising_step,
        last_mixing_denoising_step=last_mixing_denoising_step,
        device_1=device_1,
        device_2=device_2,
        save_cross_attn_weights=args.save_cross_attn_weights
        # device="balanced",
    )
    return pipe


def agent_to_EmoStory(pipe, story_path, story_json_path):
    with open(story_json_path, "r", encoding="utf-8") as f:
        story_json = json.load(f)

    args.seed = random.randint(0, 100000)
    if "seed" in story_json:
        args.seed = story_json["seed"]

    select_style = ""
    if "style" in story_json:
        select_style = story_json["style"]

    args.subject = story_json["Subject"]
    args.emotion = story_json["Emotion"]

    if "Elements" in story_json:
        args.elements = story_json["Elements"]

    args.ref_image_prompt = args.subject + f" show full body and face to the camera."
    args.story_prompts = story_json["story_prompts"]

    args.output_dir = story_path

    args.prompts = [
        f"A story of {args.ref_image_prompt} {select_style} (on top) and "
        f"{story_prompt} (on bottom)"
        for story_prompt in args.story_prompts
    ]

    print('Beginning storyboard generation...')
    os.makedirs(args.output_dir, exist_ok=True)

    print("is_enhance_elements:")
    print(args.is_enhance_elements)
    print("save_cross_attn_weights: ")
    print(args.save_cross_attn_weights)

    emostory_images, attention_store = run_mutual_story(
        pipe=pipe,
        prompts=args.prompts,
        seed=args.seed,
        guidance=args.guidance,
        n_diff_steps=args.n_diff_steps,
        same_noise=args.same_noise,
        ravm_mixing_coef=args.ravm_mixing_coef,
        first_mixing_block=args.first_mixing_block,
        last_mixing_block=args.last_mixing_block,
        first_mixing_denoising_step=args.first_mixing_denoising_step,
        last_mixing_denoising_step=args.last_mixing_denoising_step,
        is_enhance_elements=args.is_enhance_elements,
        boost_factor=args.boost_factor,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device=0 if torch.cuda.is_available() else 'cpu',
        output_dir=args.output_dir,
        args=args
    )

    save_story_images(emostory_images, output_dir=args.output_dir, json_output_dir=story_json_path, prompts=args.prompts, story_data=story_json)
    print(f'Saved EmoStory images at {args.output_dir}')


def get_theme_path_list(top_path, worker_id=0, worker_num=4):
    def check_image_exist(story_path):
        for script in os.listdir(story_path):
            if script.endswith(".png"):
                return True
        return False

    tmp_list = []

    for emotion in sorted(os.listdir(top_path)):
        emotion_path = os.path.join(top_path, emotion)
        for subject in sorted(os.listdir(emotion_path)):
            subject_path = os.path.join(emotion_path, subject)
            for theme in sorted(os.listdir(subject_path)):
                theme_path = os.path.join(subject_path, theme)
                # if check_image_exist(theme_path):
                #     print(f"{theme_path} is already have images. Skip it")
                #     continue
                tmp_list.append(theme_path)

    total = len(tmp_list)
    size = total // worker_num
    remainder = total % worker_num

    start = worker_id * size + min(worker_id, remainder)
    end = start + size + (1 if worker_id < remainder else 0)

    return tmp_list[start:end]


if __name__ == '__main__':
    args.seed = 42
    args.save_step = [3]
    #args.save_step = [3, 5, 8, 10, 15]
    agent_result_path = args.agent_result_path

    print("************Running Emostory*************")

    pipe = load_model()
    tmp = 0

    theme_path_list = get_theme_path_list(agent_result_path, worker_id=args.worker_id, worker_num=args.worker_num)

    for theme_path in tqdm(theme_path_list, desc="Generating Story"):
        story_script_path = os.path.join(theme_path, "Story_Script.json")

        tmp += 1
        print(f"currently processing the {tmp}-th story: ")
        print(story_script_path)

        agent_to_EmoStory(pipe, theme_path, story_script_path)
