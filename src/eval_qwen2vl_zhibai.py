import numpy as np
import torch
from time import time
import json
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import warnings

from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import math
import base64
from PIL import Image
import io
import pdb

# Suppress the pad_token_id warning
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    if k >= len(chunks):
        print(f"Warning: chunk_idx {k} >= num actual chunks {len(chunks)} (data_size={len(lst)}, num_chunks={n}). Returning empty list.")
        return []
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __getitem__(self, index):
        data = self.questions[index]
        inputs = safe_process_sample(self.processor, data, index)
        if inputs is None:
            # If processing failed completely, create a minimal input
            print(f"Warning: Complete processing failure for sample {index}, creating minimal input")
            textdata = {"type": "text", "text": data['messages'][0]['content'].replace("<image>", "")}
            messages = [
                {"role": "user", "content": [textdata]}
            ]
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        return inputs
    
    def collate_batch(self, instances):
        # Since batch_size=1, just return the first instance's BatchFeature
        return instances[0]

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, processor, batch_size=1, num_workers=2):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, processor)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        collate_fn=dataset.collate_batch,
        persistent_workers=False,  # Disable persistent workers to avoid memory issues
        timeout=60  # Add timeout to prevent hanging
    )
    return data_loader

def safe_process_sample(processor, data, index):
    """Safely process a sample, handling corrupted images gracefully"""
    images = data['images']
    content = []
    
    # Try to process images safely
    for i, image in enumerate(images):
        try:
            imagedata = {"type": "image", "image": image}
            content.append(imagedata)
        except Exception as e:
            print(f"Warning: Failed to process image {i} in sample {index}: {e}")
            continue
    
    textdata = {"type": "text", "text": data['messages'][0]['content'].replace("<image>", "")}
    content.append(textdata)
    messages = [
        {"role": "user", "content": content}
    ]
    
    # Try to process with images first
    max_retries = 1
    for attempt in range(max_retries):
        try:
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            return inputs
        except (OSError, ValueError, Exception) as e:
            if "truncated" in str(e) or "image file" in str(e).lower():
                print(f"Warning: Corrupted image detected in sample {index}, attempt {attempt + 1}/{max_retries}. Error: {e}")
                if attempt == max_retries - 1:
                    # On final attempt, try to process without images
                    print(f"Removing images from sample {index} due to corruption")
                    content = [textdata]  # Only keep text content
                    messages = [
                        {"role": "user", "content": content}
                    ]
                    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                    return inputs
            else:
                # Re-raise if it's not an image-related error
                return None
    
    # If all attempts fail, return None to indicate failure
    return None


def main(
        model_name,
        dev_path,
        num_chunks,
        chunk_idx,
        answers_file,
        max_seq_len = 256,
        model_fp16 = False,
        max_new_tokens = 10,
        num_workers = 2
        
):
    
    print(f"Starting evaluation with configuration:")
    print(f"  Model: {model_name}")
    print(f"  Data path: {dev_path}")
    print(f"  Num chunks: {num_chunks}, Chunk index: {chunk_idx}")
    print(f"  Num workers: {num_workers}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Output file: {answers_file}")
    
    questions = []
    with open(dev_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            questions.append(json.loads(line.strip()))
    print("data_size: %d" % len(questions))
    questions = get_chunk(questions, num_chunks, chunk_idx)
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Load InternVL3 model and processor
    print("model name:", model_name)
    
    # Load processor for InternVL3
    processor = AutoProcessor.from_pretrained(
        model_name,
        eos_token_id='<|endoftext|>',
        pad_token='<|endoftext|>',
        trust_remote_code=True
    )
    
    # Ensure pad token is set for the tokenizer
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    if model is not None:
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        model.generation_config.top_k=None
    
    # Ensure model's config has pad token set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    
    # Also set pad_token_id in the tokenizer config
    if hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    # Set model to evaluation mode
    model.eval()

    # Get the model's dtype for consistent tensor creation
    model_dtype = next(model.parameters()).dtype

    data_loader = create_data_loader(questions, processor, num_workers=num_workers)

    choices = ['否', '是']
    token_idx = [processor.tokenizer.encode(label, add_special_tokens=False)[0] for label in choices]
    assert(len(choices) == len(set(token_idx)))
    print(choices, token_idx)

    indx=0
    for inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        try:
            idx = line["id"]
            
            # Move inputs to GPU and convert to model's dtype
            inputs = inputs.to(model.device, dtype=model_dtype)
            
            # Inference: Generation of the output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    min_length=0, 
                    num_beams=1, 
                    num_return_sequences=1,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    use_cache=False,
                    do_sample=False,
                    output_scores=True,
                    eos_token_id=model.config.eos_token_id,
                    pad_token_id=model.config.pad_token_id
                )
            
            # Extract generated sequences and scores
            generated_ids = outputs.sequences
            logits = outputs.scores
            
            # Calculate probabilities for the choice tokens
            if logits and len(logits) > 0:
                # Get the first token's logits (for next token prediction)
                first_logits = logits[0]
                # Extract logits for choice tokens
                choice_logits = torch.stack([first_logits[:, i] for i in token_idx], dim=1)
                probs = torch.softmax(choice_logits, dim=-1)
                probs = probs.squeeze().cpu().numpy()
            else:
                print("failed ", line)
                continue

            # Decode generated text - inputs is a BatchFeature object
            input_ids = inputs.input_ids
                
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = processor.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Convert probabilities to list
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
            elif isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy().tolist()
            
            results = {
                "question_id": idx,
                "text": output_text,
                "score": probs,
                "metadata": {}
            }
            
            ans_file.write(json.dumps(results, ensure_ascii=False) + "\n")
            indx = indx + 1
            ans_file.flush()
            
        except Exception as e:
            print(f"Error processing sample {indx}: {e}")
            print(f"Sample data: {line}")
            continue
    
    ans_file.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Used for result dir name",
        required=True,
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="",
        help="path of training dataset",
        required=True,
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=5,
        help="max tokens for generation",
    )
    parser.add_argument(
        "--chunk_idx",
        type=int,
        default=0,
        help="max tokens for generation",
    )
    parser.add_argument(
        "--ans_file",
        type=str,
        default="",
        help="",
        required=True,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=830,
        help="",
    )
    parser.add_argument(
        "--model_fp16",
        action="store_true",
        help="Using fp16 for models",
    )
    parser.add_argument(
        "--max_new_token",
        type=int,
        default=5,
        help="max tokens for generation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for data loading (0 for single process)",
    )
    args = parser.parse_args()

    main(
        args.model_name,
        args.dev_path,
        args.num_chunks,
        args.chunk_idx,
        args.ans_file,
        args.max_seq_len,
        args.model_fp16,
        args.max_new_token,
        args.num_workers,
    )
