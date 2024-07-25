import gradio as gr
import time
from main import main
import os
from transformers import AutoTokenizer, TextIteratorStreamer
from ipex_llm.transformers import AutoModelForCausalLM
import torch
from threading import Thread, Event

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "8"  # 设置OpenMP线程数为8,用于控制并行计算

import nest_asyncio
import asyncio

#nest_asyncio.apply()
#loop = asyncio.new_event_loop()
#asyncio.set_event_loop(loop)    


# 定义用户输入处理函数
def user(user_message, history):
    return "", history + [[user_message, None]]  # 返回空字符串和更新后的历史记录

# 定义机器人回复生成函数
async def bot(history):
    #stop_event.clear()  # 重置停止事件
    prompt = history[-1][0]  # 获取最新的用户输入
    
    
    start_time = time.time()  # 记录开始时间
    print(f"\n用户输入: {prompt}")
    print("\n模型输出: ", )
    

    generated_text = await main(prompt)
    
    #print(generated_text)
    history[-1][1] = generated_text  # 更新历史记录中的回复
    yield history  # 逐步返回更新的历史记录

    end_time = time.time()
    print(f"\n\n生成完成，用时: {end_time - start_time:.2f} 秒")

# 定义停止生成函数
#def stop_generation():
    #stop_event.set()  # 设置停止事件

# 使用Gradio创建Web界面
with gr.Blocks() as demo:
    gr.Markdown("# Qwen 聊天机器人")
    chatbot = gr.Chatbot()  # 聊天界面组件
    msg = gr.Textbox()  # 用户输入文本框
    clear = gr.Button("清除")  # 清除按钮
    #stop = gr.Button("停止生成")  # 停止生成按钮
    #myrag = Myrag()
    

    # 设置用户输入提交后的处理流程
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)  # 清除按钮功能
    #stop.click(stop_generation, queue=False)  # 停止生成按钮功能

if __name__ == "__main__":
    print("启动 Gradio 界面...")
    demo.queue()  # 启用队列处理请求
    
    #name = os.environ['JUPYTER_NAME']
    #region = os.environ['dsw_region']

    #host = "dsw-gateway-{region}.data/aliyun.com".format(region=region)

    port = 7860
    #root_path = f'/{name}/proxy/{port}'

    #demo.launch(root_path=root_path, server_port=port)# 兼容魔搭情况下的路由
    demo.launch()
