from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2, torch
import numpy as np
from PIL import Image
import os, math, hashlib, requests
from decord import VideoReader, cpu
import json, re, textwrap, os, datetime
from typing import List, Dict
from collections import Counter
from utils import TASKS

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image

def inference(processor, model, video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def inference_from_frames(frames_np, prompt, processor, model, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28, temperature=0.7):
    """
    frames_np: a NumPy array of shape (N, H, W, 3), dtype=uint8, RGB.
    prompt:    the text prompt (string).
    processor: AutoProcessor (Qwen's processor).
    model:     Qwen2_5_VLForConditionalGeneration model.
    """
    pil_list = [Image.fromarray(fr) for fr in frames_np]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {
                "type": "video",
                "video": pil_list,
                "total_pixels": total_pixels,
                "min_pixels": min_pixels,
            },
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=video_kwargs["fps"], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
 
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature) 

    generated_ids = [
        output_ids[i, inputs.input_ids.shape[-1]:]
        for i in range(output_ids.shape[0])
    ]

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def _extract_json_and_sentence(text: str, config) -> Dict:
    """
    Robustly extracts, cleans, and restructures VLM output to a flat dictionary.
    """
    final_data = {
        "outcome": "failure",
        "primary_error": {"code": "unknown", "explanation": "VLM output was malformed."},
        "secondary_factors": [],
        "key_frame_indices": [],
        "suggested_fix": TASKS[config.env_name],
        "confidence": 0.0,
        "summary": TASKS[config.env_name]
    }

    try:
        cleaned_text = text.replace("```json", "").replace("```", "").strip()
        json_start_index = cleaned_text.find('{')
        json_end_index = cleaned_text.rfind('}')

        if json_start_index != -1 and json_end_index != -1:
            json_string = cleaned_text[json_start_index : json_end_index + 1]
            parsed_vlm_json = json.loads(json_string)
            
            source_dict = parsed_vlm_json.get('properties', parsed_vlm_json)
            final_data["outcome"] = source_dict.get("outcome", final_data["outcome"])
            final_data["primary_error"] = source_dict.get("primary_error", final_data["primary_error"])
            final_data["secondary_factors"] = source_dict.get("secondary_factors", final_data["secondary_factors"])
            final_data["key_frame_indices"] = source_dict.get("key_frame_indices", final_data["key_frame_indices"])
            final_data["suggested_fix"] = source_dict.get("suggested_fix", final_data["suggested_fix"])
            final_data["confidence"] = source_dict.get("confidence", final_data["confidence"])

            summary_string = cleaned_text[json_end_index + 1:].strip()
            final_data['summary'] = summary_string if summary_string else "No summary sentence provided."

    except Exception as e:

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            final_data["summary"] = lines[-1]

    return final_data

# --- Adding Historical Context ---
# Goal: Feed the model a tiny digest of the last N episodes for this task so it learns from trends.
def _load_recent_history(memory_path: str, task: str, k: int = 5) -> List[Dict]:
    try:
        if not os.path.exists(memory_path): return []
        recs = []
        with open(memory_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if not isinstance(rec.get("outcome"), str):
                        continue  # ← Skip schema-style records
                    if rec.get("task") == task:
                        recs.append(rec)
                except Exception:
                    pass
        return recs[-k:]
    except Exception:
        return []

def _append_memory(memory_path: str, record: Dict):
    # Skip malformed records (e.g., missing outcome string)
    if not isinstance(record.get("outcome"), str):
        print("[Warning] Skipping invalid feedback record (schema?)")
        return
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    with open(memory_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _summarize_history_for_prompt(records: List[Dict]) -> str:
    if not records:
        return "No prior episodes."
    # compact textual digest: outcome counts + recurring primary_error codes + typical fixes
    
    outs = Counter(r.get("outcome") for r in records)
    errs = Counter((r.get("primary_error") or {}).get("code","unknown") for r in records)
    top_errs = ", ".join([f"{k}×{v}" for k,v in errs.most_common(3)])
    top_fixes = [r.get("suggested_fix","") for r in records if r.get("suggested_fix")]
    tip = top_fixes[-1] if top_fixes else ""
    return f"Recent {len(records)} episodes — outcomes: {dict(outs)}; frequent errors: {top_errs or 'none'}. Last suggested fix: {tip or 'n/a'}."



def _log_invalid_feedback(raw_text: str, parsed_dict: dict, exp_name: str = None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/invalid_feedback"
    os.makedirs(log_dir, exist_ok=True)
    file_name = f"{exp_name or 'feedback'}_{timestamp}.txt"
    with open(os.path.join(log_dir, file_name), "w") as f:
        f.write("=== RAW OUTPUT ===\n")
        f.write(raw_text.strip() + "\n\n")
        f.write("=== PARSED ===\n")
        f.write(json.dumps(parsed_dict, indent=2, ensure_ascii=False))