from ipex_llm.transformers import AutoModelForCausalLM
from transformers import  AutoTokenizer
import os
if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(),"qwen2chat_src/Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int8', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.save_low_bit('qwen2chat_7b_int8')
    tokenizer.save_pretrained('qwen2chat_7b_int8')