import numpy as np
import torch
from time import time
import json
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import math
#import pdb

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer):
        self.questions = questions
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.questions[index]
        if "instruction" in data and "input" in data:
            prompt = "\n".join([data["instruction"], data["input"]])
        elif "input" in data:
            prompt = data["input"]
        elif "messages" in data:
            prompt = data['messages'][0]['content'].replace("<image>", "")
        else:
            print("invalid format", data)
            prompt = ""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ] 
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # print("messages: ", messages, "\ntext: ", text)
        input_ids = self.tokenizer([text], return_tensors="pt")
        data_dict = dict(input_ids=input_ids)
        return data_dict
    
    def collate_batch(self, instances):
        input_ids = [instance['input_ids'] for instance in instances]
        batch = dict(
            input_ids=input_ids
        )
        return batch['input_ids']

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, tokenizer, batch_size=1, num_workers=7):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,collate_fn=dataset.collate_batch)
    return data_loader

def main(
        model_name,
        dev_path,
        num_chunks,
        chunk_idx,
        answers_file,
        max_seq_len = 256,
        model_fp16 = False,
        max_new_tokens = 10
        
):
    
    questions = []
    # 读取json格式的文件
    with open(dev_path, "r", encoding='utf-8') as f:
        if dev_path.endswith("json"):
            questions = json.load(f)
        elif dev_path.endswith("jsonl"):
            lines = f.readlines()
            for line in lines:
                questions.append(json.loads(line.strip()))
        else:
            print("invalid data")
            quit()
    print("data_size: %d" % len(questions))
    questions = get_chunk(questions, num_chunks, chunk_idx)
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding='utf-8')

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    print("model name:", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    if model is not None:
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        model.generation_config.top_k=None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token='<|endoftext|>')
    data_loader = create_data_loader(questions, tokenizer)

    choices = ['否', '是']
    token_idx = [tokenizer.encode(label, add_special_tokens=False)[0] for label in choices]
    assert(len(choices) == len(set(token_idx)))
    print(choices, token_idx)

    indx=0
    for inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        # pdb.set_trace()
        idx = line["id"]
        inputs = inputs[0].to(device='cuda', non_blocking=True)
        # Inference: Generation of the output
        # pdb.set_trace()
        outputs = model.generate(**inputs,
                                 min_length=0, num_beams=1, num_return_sequences=1,
                                 max_new_tokens=max_new_tokens,
                                 # 重复惩罚核心参数（关键）
                                 repetition_penalty=1.2,        # 重复惩罚因子，1.2是常用合理值
                                 no_repeat_ngram_size=2,        # 禁止重复2个连续字/词
                                 return_dict_in_generate=True,
                                 use_cache=False,
                                 do_sample=False,
                                 output_scores=True)
        # pdb.set_trace()
        generated_ids=outputs.sequences
        logits = outputs.scores
        # print("logprobs: ", torch.cat([logits[0][:, i] for i in token_idx]))
        probs = torch.softmax(torch.cat([logits[0][:, i] for i in token_idx], 0), dim=-1) 

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        probs=[float(i) for i in probs]
        results =  {"question_id": idx,
                    "text": output_text,
                    "score": probs,
                    "metadata": {}}
        # print(results)
        ans_file.write(json.dumps(results, ensure_ascii=False) + "\n")
        indx=indx+1
        ans_file.flush()
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
        default=10240,
        help="",
    )
    parser.add_argument(
        "--model_fp16",
        action="store_true",
        help="Using fp16 for models",#默认为false
    )
    parser.add_argument(
        "--max_new_token",
        type=int,
        default=1,
        help="max tokens for generation",
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
    )
