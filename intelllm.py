# 设置OpenMP线程数为8
import os
import time
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from typing import Any, List, Optional

# 从llama_index库导入所需的类
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ipex_llm.llamaindex.llms import IpexLLM

from modelscope import snapshot_download
MODEL_BASE_DIR = 'models'
llm_model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir=MODEL_BASE_DIR, revision='master')
emb_model_dir = snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir=MODEL_BASE_DIR, revision='master')
#reranker_model_dir = snapshot_download('BAAI/bge-reranker-base', cache_dir=MODEL_BASE_DIR, revision='master')

q_llm_model_dir = "qwen25-1.5b_int8"

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import  AutoTokenizer
def prepare_quantization():
    # Check if the target directory exists
    if not os.path.exists(q_llm_model_dir):
        model_path = os.path.join(os.getcwd(), llm_model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int8', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    

        # If not, save the model and tokenizer to the target directory
        model.save_low_bit(q_llm_model_dir)
        tokenizer.save_pretrained(q_llm_model_dir)
        print(f"Model and tokenizer saved to {q_llm_model_dir}")
    else:
        print(f"Directory {q_llm_model_dir} already exists. Skipping save.")


class Config:
    """配置类,存储所有需要的参数"""
    model_path = q_llm_model_dir
    tokenizer_path = q_llm_model_dir
    #question = "How does Llama 2 perform compared to other open-source models?"
    #data_path = "./data/llamatiny.pdf"
    #persist_dir = "./chroma_db"
    embedding_model_path = emb_model_dir
    max_new_tokens = 512

def completion_to_prompt(completion: str) -> str:
    """
    将完成转换为提示格式
    
    Args:
        completion (str): 完成的文本
    
    Returns:
        str: 格式化后的提示
    """
    return f"\n</s>\n\n{completion}</s>\n\n"

def messages_to_prompt(messages: List[dict]) -> str:
    """
    将消息列表转换为提示格式
    
    Args:
        messages (List[dict]): 消息列表
    
    Returns:
        str: 格式化后的提示
    """
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"\n{message['content']}</s>\n"
        elif message["role"] == "user":
            prompt += f"\n{message['content']}</s>\n"
        elif message["role"] == "assistant":
            prompt += f"\n{message['content']}</s>\n"

    if not prompt.startswith("\n"):
        prompt = "\n</s>\n" + prompt

    prompt = prompt + "\n"

    return prompt

def setup_llm(config: Config) -> IpexLLM:
    """
    设置语言模型
    
    Args:
        config (Config): 配置对象
    
    Returns:
        IpexLLM: 配置好的语言模型
    """
    return IpexLLM.from_model_id_low_bit(
        model_name=config.model_path,
        tokenizer_name=config.tokenizer_path,
        context_window=4096,
        max_new_tokens=config.max_new_tokens,
        generate_kwargs={
            "temperature": 0.7,  # 控制生成的多样性
            "do_sample": True,  # 采样
            "top_p": 0.85,  # Top-p 采样设置，限制多样性
            "top_k": 50,  # Top-k 采样限制
            "repetition_penalty": 1.2,  # 重复惩罚
            #"max_length": 512  # 限制最大生成长度
        },
        model_kwargs={},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="cpu",
    )
from llama_index.core.postprocessor import SentenceTransformerRerank
class Intelllm:

    def __init__(self) -> None:
        prepare_quantization()
        self.config = Config()
        self.embed_model = HuggingFaceEmbedding(model_name=self.config.embedding_model_path)
        self.llm = setup_llm(self.config)
        #self.reranker = SentenceTransformerRerank(
        #    top_n=5,
        #    model=reranker_model_dir
        #)

