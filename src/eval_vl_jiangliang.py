"""Qwen3-VL 多模态推理脚本（适配 pdpl 测试集 conversations 格式）。

功能要点：
1. 使用 AutoProcessor + AutoModelForImageTextToText，自动适配 Qwen3-VL / Qwen2.5-VL。
2. 兼容数据三种字段格式：messages / conversations(ShareGPT) / instruction+input。
3. 图像字段兼容：images(list) / image(list 或 str)。
4. 二分类是/否 logits 概率提取，与 convert_vqav2_for_submission_qwen3_vl_jiangliang.py 对齐。
"""
import argparse
import json
import math
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

# 屏蔽部分无关警告，避免日志被刷屏
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")


def load_processor(model_name):
    """健壮地加载 Qwen-VL 系列 processor。

    在某些 transformers 版本下，AutoProcessor.from_pretrained 可能错误地
    退化为 tokenizer（返回 Qwen2TokenizerFast）。这里按优先级依次尝试：
    1) Qwen3VLProcessor（>=4.57 才有）
    2) AutoProcessor（必须真正返回 multimodal processor）
    3) Qwen2_5_VLProcessor（旧环境下的兼容降级）
    4) Qwen2VLProcessor（更老环境下的兜底降级）
    任一成功即返回。
    """
    last_err = None

    def _try(name, loader):
        nonlocal last_err
        try:
            proc = loader()
            # 必须是真正的 multimodal processor，含有 image_processor 才算成功
            if not hasattr(proc, "image_processor"):
                raise RuntimeError(
                    f"{name} returned a non-multimodal object: {type(proc).__name__}"
                )
            print(f"[info] processor loaded via {name}: {type(proc).__name__}")
            return proc
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[warn] {name} load failed: {e}")
            return None

    # 1) 显式 Qwen3VLProcessor
    def _load_qwen3vl():
        from transformers import Qwen3VLProcessor  # type: ignore
        return Qwen3VLProcessor.from_pretrained(model_name, trust_remote_code=True)

    proc = _try("Qwen3VLProcessor", _load_qwen3vl)
    if proc is not None:
        return proc

    # 2) AutoProcessor
    def _load_auto():
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    proc = _try("AutoProcessor", _load_auto)
    if proc is not None:
        return proc

    # 3) Qwen2.5-VL processor（与 Qwen3-VL checkpoint 文件结构基本兼容）
    def _load_qwen25vl():
        from transformers import Qwen2_5_VLProcessor  # type: ignore
        return Qwen2_5_VLProcessor.from_pretrained(model_name, trust_remote_code=True)

    proc = _try("Qwen2_5_VLProcessor", _load_qwen25vl)
    if proc is not None:
        return proc

    # 4) Qwen2-VL processor（最老兜底）
    def _load_qwen2vl():
        from transformers import Qwen2VLProcessor  # type: ignore
        return Qwen2VLProcessor.from_pretrained(model_name, trust_remote_code=True)

    proc = _try("Qwen2VLProcessor", _load_qwen2vl)
    if proc is not None:
        return proc

    raise RuntimeError(
        f"Failed to load processor for {model_name}: {last_err}. "
        f"Hint: please use a python env with transformers>=4.57 that contains Qwen3VLProcessor."
    )


def get_tokenizer(processor):
    """从 processor 中安全获取 tokenizer。兼容 processor 本身即 tokenizer 的情况。"""
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return tok
    # processor 本身就是 tokenizer
    if hasattr(processor, "encode") and hasattr(processor, "decode"):
        return processor
    raise AttributeError("processor has no tokenizer attribute and is not a tokenizer itself")


def split_list(lst, n):
    """把列表均匀切成 n 块"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def extract_prompt_and_images(data):
    """从一条样本中抽取出 prompt 文本与图像路径列表。

    支持以下数据格式：
    - {"instruction": str, "input": str, "image"/"images": list/str}
    - {"messages": [{"role": "user", "content": str}, ...], "image"/"images": list/str}
    - {"conversations": [{"from": "human", "value": str}, ...], "image"/"images": list/str}
    """
    # 1) 文本 prompt
    if "instruction" in data and "input" in data:
        prompt = "\n".join([data["instruction"], data["input"]])
    elif "messages" in data and len(data["messages"]) > 0:
        prompt = data["messages"][0].get("content", "")
    elif "conversations" in data and len(data["conversations"]) > 0:
        # ShareGPT 格式：human 是输入
        first = data["conversations"][0]
        prompt = first.get("value", first.get("content", ""))
    else:
        print("invalid format", data)
        prompt = ""

    # 移除 <image> 占位符（apply_chat_template 会自动加 vision token）
    prompt = prompt.replace("<image>", "")

    # 2) 图像
    images = []
    if "images" in data and data["images"]:
        images = data["images"] if isinstance(data["images"], list) else [data["images"]]
    elif "image" in data and data["image"]:
        images = data["image"] if isinstance(data["image"], list) else [data["image"]]

    # 过滤掉缺失文件，避免 processor 报错
    valid_images = [img for img in images if isinstance(img, str) and os.path.isfile(img)]
    return prompt, valid_images


class CustomDataset(Dataset):
    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor

    def __getitem__(self, index):
        data = self.questions[index]
        inputs = safe_process_sample(self.processor, data, index)
        if inputs is None:
            # 兜底：纯文本输入
            prompt, _ = extract_prompt_and_images(data)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                template_kwargs={"enable_thinking": False},
                return_dict=True,
                return_tensors="pt",
            )
        return inputs

    def collate_batch(self, instances):
        # batch_size=1，直接返回首个 BatchFeature
        return instances[0]

    def __len__(self):
        return len(self.questions)


def safe_process_sample(processor, data, index):
    """构造 chat 模板输入，遇到坏图自动降级为纯文本。"""
    prompt, images = extract_prompt_and_images(data)

    content = []
    for img_path in images:
        content.append({"type": "image", "image": img_path})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            template_kwargs={"enable_thinking": False},
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
    except (OSError, ValueError, Exception) as e:  # noqa: BLE001
        msg = str(e).lower()
        if "truncated" in msg or "image" in msg or "decode" in msg:
            print(f"[warn] sample {index} image error -> fallback to text-only: {e}")
            try:
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                return processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    template_kwargs={"enable_thinking": False},
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as e2:  # noqa: BLE001
                print(f"[error] sample {index} fallback failed: {e2}")
                return None
        print(f"[error] sample {index} processing failed: {e}")
        return None


def create_data_loader(questions, processor, batch_size=1, num_workers=2):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, processor)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dataset.collate_batch,
        persistent_workers=False,
        timeout=120 if num_workers > 0 else 0,
    )
    return data_loader


def load_questions(dev_path):
    """同时支持 json / jsonl 输入。"""
    questions = []
    try:
        with open(dev_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
            if not isinstance(questions, list):
                raise ValueError("not a list")
    except Exception:
        questions = []
        with open(dev_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return questions


def main(
    model_name,
    dev_path,
    num_chunks,
    chunk_idx,
    answers_file,
    max_seq_len=256,
    model_fp16=False,
    max_new_tokens=10,
    num_workers=2,
):
    print("Starting evaluation with configuration:")
    print(f"  Model: {model_name}")
    print(f"  Data path: {dev_path}")
    print(f"  Num chunks: {num_chunks}, Chunk index: {chunk_idx}")
    print(f"  Num workers: {num_workers}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Output file: {answers_file}")

    # 环境自检：打印关键依赖来源，避免子进程从其他 conda env 的 site-packages 加载到旧版包
    import sys as _sys
    print(f"  Python: {_sys.executable}")
    try:
        import transformers as _tf
        print(f"  transformers: {_tf.__version__} @ {_tf.__file__}")
    except Exception as _e:  # noqa: BLE001
        print(f"  [warn] transformers import error: {_e}")
    try:
        import mistral_common as _mc
        print(f"  mistral_common: {_mc.__file__}")
    except Exception as _e:  # noqa: BLE001
        print(f"  [info] mistral_common not available: {_e}")

    questions = load_questions(dev_path)
    print(f"data_size: {len(questions)}")
    questions = get_chunk(questions, num_chunks, chunk_idx)

    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    # 加载 processor / model
    print("model name:", model_name)
    processor = load_processor(model_name)
    print("processor type:", type(processor).__name__)
    tokenizer = get_tokenizer(processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 注意：Qwen3-VL checkpoint 的 config.json 里子模块 dtype 混合声明
    # （顶层 bfloat16，部分视觉子模块 float32），torch_dtype="auto" 会沿用这种
    # 不一致设置，前向到 lm_head 时会触发 "float != c10::BFloat16"。
    # 这里显式强制全模型 bfloat16，规避 dtype 不一致问题。
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if model.generation_config is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    model_dtype = next(model.parameters()).dtype

    data_loader = create_data_loader(questions, processor, num_workers=num_workers)

    # 二分类标签：与训练 prompt「请只用是或者否来回答」保持一致
    choices = ["否", "是"]
    token_idx = [tokenizer.encode(label, add_special_tokens=False)[0] for label in choices]
    assert len(choices) == len(set(token_idx)), f"choice token collide: {choices} -> {token_idx}"
    print("choices:", choices, "token_idx:", token_idx)

    indx = 0
    _err_dump_left = 3  # 仅对前几条出错样本打印完整 traceback，避免日志爆炸
    for inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        try:
            idx = line["id"]

            # 把张量挪到 device
            inputs = inputs.to(model.device)
            # 把 BatchFeature 中所有浮点张量统一转为 model_dtype（如 bf16），
            # 避免 ViT/Linear 因 fp32 输入与 bf16 权重不匹配而报 dtype 错误。
            # int 类张量（input_ids / attention_mask / *_grid_thw 等）保持原样。
            for _k, _v in list(inputs.items()):
                if isinstance(_v, torch.Tensor) and _v.is_floating_point():
                    inputs[_k] = _v.to(dtype=model_dtype)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    min_length=0,
                    num_beams=1,
                    num_return_sequences=1,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    use_cache=True,
                    do_sample=False,
                    output_scores=True,
                    eos_token_id=model.config.eos_token_id,
                    pad_token_id=model.config.pad_token_id,
                )

            generated_ids = outputs.sequences
            logits = outputs.scores

            if not logits:
                print("failed (no logits)", line.get("id"))
                continue

            first_logits = logits[0]  # (1, vocab)
            choice_logits = torch.stack([first_logits[:, i] for i in token_idx], dim=1)
            probs = torch.softmax(choice_logits, dim=-1).squeeze(0).detach().cpu().numpy()

            input_ids = inputs.input_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
            elif isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy().tolist()

            results = {
                "question_id": idx,
                "text": output_text,
                "score": probs,
                "metadata": {},
            }
            ans_file.write(json.dumps(results, ensure_ascii=False) + "\n")
            indx += 1
            ans_file.flush()
        except Exception as e:  # noqa: BLE001
            if _err_dump_left > 0:
                _err_dump_left -= 1
                import traceback as _tb
                print(f"\n=== Error processing sample {indx} (id={line.get('id')}): {e} ===")
                try:
                    _info = {}
                    for _k, _v in inputs.items():
                        if isinstance(_v, torch.Tensor):
                            _info[_k] = f"{tuple(_v.shape)} {_v.dtype} {_v.device}"
                        else:
                            _info[_k] = type(_v).__name__
                    print(f"    inputs: {_info}")
                    print(f"    model_dtype: {model_dtype}")
                except Exception:  # noqa: BLE001
                    pass
                _tb.print_exc()
            else:
                print(f"Error processing sample {indx} (id={line.get('id')}): {e}")
            continue

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="模型路径或 HF id")
    parser.add_argument("--dev_path", type=str, required=True, help="测试集 json/jsonl 路径")
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--ans_file", type=str, required=True, help="输出 jsonl 路径")
    parser.add_argument("--max_seq_len", type=int, default=830)
    parser.add_argument("--model_fp16", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    main(
        args.model_name,
        args.dev_path,
        args.num_chunks,
        args.chunk_idx,
        args.ans_file,
        args.max_seq_len,
        args.model_fp16,
        args.max_new_tokens,
        args.num_workers,
    )
