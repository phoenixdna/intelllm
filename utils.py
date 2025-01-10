import base64
import requests
import os


def covert_image_to_base64(image_path):
    # 获得文件后缀名
    ext = image_path.split(".")[-1]
    if ext not in ["gif", "jpeg", "png"]:
        ext = "jpeg"

    with open(image_path, "rb") as image_file:
        # Read the file
        encoded_string = base64.b64encode(image_file.read())

        # Convert bytes to string
        base64_data = encoded_string.decode("utf-8")

        # 生成base64编码的地址
        base64_url = f"data:image/{ext};base64,{base64_data}"
        return base64_url



def format_welcome_html():
    config = {
        'name': "法律小助手",
        'description': '系统集成刑法和民法的最新条款，结合Qwen2.5模型的强大语言理解和生成能力以及使用ipex-llm的低成本推理能力，提供精确的法律咨询服务和解读分析。😊',
        'introduction_label': "<br>软件功能",
        'rule_label': "<br>操作指南",
        'char1': 'RAG技术实现高效精准的法律条文检索，帮助用户快速获取所需信息，大幅提升法律研究效率。',
        'char2': '利用大模型的对话能力，对法律案件进行深入分析与初步判定，为用户提供智能化法律建议，简化复杂的法律咨询流程。',
        'char3': '通过Intel IPEX技术优化CPU推理性能，使用户在降低成本的同时，享受到流畅且高效的法律服务体验。',

        'rule1': '1.法律咨询：在主界面，用户可通过自然语言与系统对话，系统将根据用户的问题进行法律案件分析，并提供专业化的法律咨询建议。',
        'rule2': '2.法律检索：系统会匹配用户输入的问题并检索出Top5的法律条文以及相应的相关度评分，帮助用户快速匹配问题相关的法律条文信息，进行高效的法律检索',
        'rule3': '3.法律原文阅读：在知识库中，用户可以查阅已经索引的法律原文。目前演示系统已加载刑法和民法的相关内容，用户可以随时查阅',

    }
    image_src = covert_image_to_base64('logo1.jpg')
    return f"""
<div class="bot_cover">
    <div class="bot_avatar">
        <img src={image_src} />
    </div>
    <div class="bot_name">{config.get("name")}</div>
    <div class="bot_desc">{config.get("description")}</div>
    <div class="bot_intro_label">{config.get("introduction_label")}</div>
    <div class="bot_intro_ctx">
        <ul>
            <li>{config.get("char1")}</li>
            <li>{config.get("char2")}</li>
            <li>{config.get("char3")}</li>
            
        </ul>
    </div>
    <div class="bot_intro_label">{config.get("rule_label")}</div>
    <div class="bot_intro_ctx">
        <ul>
            <li>{config.get("rule1")}</li>
            <li>{config.get("rule2")}</li>
            <li>{config.get("rule3")}</li>
        </ul>
    </div>
</div>
"""
