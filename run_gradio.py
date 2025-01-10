import gradio as gr
import time
from main import Main
import pandas as pd
import os
from transformers import AutoTokenizer, TextIteratorStreamer
from ipex_llm.transformers import AutoModelForCausalLM
import torch
from threading import Thread, Event

from utils import format_welcome_html
from pdfminer.high_level import extract_text

mymain = Main()

# 定义用户输入处理函数
def user(user_message, history):
    return "", history + [[user_message, None]]  # 返回空字符串和更新后的历史记录

# 定义机器人回复生成函数
async def bot(history):
    global mymain

    await mymain.setup()  # 等待mymain.setup()异步执行
    #stop_event.clear()  # 重置停止事件
    prompt = history[-1][0]  # 获取最新的用户输入
    
    
    start_time = time.time()  # 记录开始时间
    print(f"\n用户输入: {prompt}")
    #text = await mymain.myrag(prompt)

    generated_text = await mymain.main(prompt)
    
    #print(generated_text)
    history[-1][1] = generated_text  # 更新历史记录中的回复
    yield history  # 逐步返回更新的历史记录

    end_time = time.time()
    print(f"\n\n大模型推理生成完成，用时: {end_time - start_time:.2f} 秒")

# 定义机器人回复生成函数
async def get_refer(history):
    prompt = history[-1][0]  # 获取最新的用户输入
    global mymain
    await mymain.setup()  # 等待mymain.setup()异步执行
    start_time = time.time()  # 记录开始时间
    text = await mymain.myrag(prompt)
    end_time = time.time()
    print(f"\n\nRAG检索完成，用时: {end_time - start_time:.2f} 秒")
    return text

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def game_ui():
    return {tabs: gr.update(visible=False), main_tabs: gr.update(visible=True)}


def welcome_ui():
    return {tabs: gr.update(visible=True), main_tabs: gr.update(visible=False)}

# 定义停止生成函数
#def stop_generation():
    #stop_event.set()  # 设置停止事件

# 读取指定目录下的PDF文件列表
def list_pdfs(directory):
    return [f for f in os.listdir(directory) if f.endswith('.pdf')]

# 提取PDF中的文本信息
def extract_pdf_text(pdf_path):
    return extract_text(pdf_path)


# 更新文本框内容
def update_textbox(selected_file, directory):
    if selected_file:
        pdf_path = os.path.join(directory, selected_file)
        return extract_pdf_text(pdf_path)
    return ""


# 文件目录路径
directory = "data"

# 获取目录下的PDF文件列表
pdf_files = list_pdfs(directory)

# 转换为DataFrame格式
pdf_df = pd.DataFrame(pdf_files, columns=["PDF文件名"])



# 使用Gradio创建Web界面
with gr.Blocks(js=js_func,css='appBot.css') as demo:
    demo.title = 'LawAssistant'
    gr.Markdown('''<center><font size=6 style="color: #FFFFFF;">法律小助手V0.7</font></center>''')
    tabs = gr.Tabs(visible=True)
    with tabs:
        welcome_tab = gr.Tab('软件介绍', id=0)
        with welcome_tab:
            user_chat_bot_cover = gr.HTML(format_welcome_html())

        with gr.Row():
            new_button = gr.Button(value='🚀开始法律咨询', variant='primary')

    main_tabs = gr.Tabs(visible=False)
    with main_tabs:
        with gr.Tab('主界面'):
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="resizable-chatbot")  # 聊天界面组件
                refer_output = gr.Textbox(label="检索结果", lines=10, interactive=False, visible=True)
            msg = gr.Textbox(label= "用户输入", placeholder='请在此输入您的问题，回车提交')  # 用户输入文本框
            with gr.Column():
                clear = gr.Button("清除")  # 清除按钮

        with gr.Tab('知识库界面'):
            with gr.Row():
                # 左侧直接显示的文件列表
                with gr.Column():
                    file_list = gr.Dataframe(
                        value=pdf_df,
                        headers=["PDF文件名"],
                        type="array",
                        label="PDF文件列表",
                        interactive=False
                    )

                # 右侧文本框
                with gr.Column():
                    text_box = gr.Textbox(
                        label="PDF内容",
                        lines=20,
                        elem_id="pdf_textbox",
                        visible=True
                    )
        with gr.Row():
            return_welcome_button = gr.Button(value="↩️返回首页")
        #stop = gr.Button("停止生成")  # 停止生成按钮
        #myrag = Myrag()




    # 设置用户输入提交后的处理流程
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(get_refer, chatbot, refer_output
    ).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: ([], "",""), None, [chatbot, refer_output,text_box], queue=False)  # 清除按钮功能
    #stop.click(stop_generation, queue=False)  # 停止生成按钮功能

    # 3. 绑定异步回调函数


    new_button.click(game_ui, outputs=[tabs, main_tabs])
    return_welcome_button.click(welcome_ui, outputs=[tabs, main_tabs])

    # 更新文本框内容
    def on_select_file(selected_row):
        if selected_row:
            selected_file = selected_row[0][0]  # 提取选中的文件名
            return update_textbox(selected_file, directory)
        return ""


    file_list.select(
        fn=on_select_file,
        inputs=file_list,
        outputs=text_box
    )

if __name__ == "__main__":
    print("启动 Gradio 界面...")
    demo.queue()  # 启用队列处理请求
    
    #name = os.environ['JUPYTER_NAME']
    #region = os.environ['dsw_region']

    #host = "dsw-gateway-{region}.data/aliyun.com".format(region=region)

    port = 7860
    #root_path = f'/{name}/proxy/{port}'

    #demo.launch(root_path=root_path, server_port=port)# 兼容魔搭情况下的路由
    demo.launch(server_name='0.0.0.0', server_port=port)
