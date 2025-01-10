# 法律小助手

## 项目简介
本项目基于 IPEX-LLM 和llamaindex技术开发，旨在通过纯CPU进行模型推理，为用户提供法律相关问题的智能解答服务。
GUI界面通过解析用户输入问题，引用相关法律条文进行回答，适用于法律咨询等场景。
目前实测性能为：
- 2个文档268个节点，大约0.05秒RAG搜索时间；（采用了bge-small-zh-v1.5）
- 模型CPU推理时间大约20-30秒；（采用的是Qwen2.5-1.5B-Instruct的Int8 量化）

## 推荐软件运行环境要求
- **CPU**: 6核及以上
- **内存**: 16GB及以上
- **Python 版本**: 3.9+
- **Torch版本**：2.2.2+

## 项目目录结构
```
│  .env                             # 环境变量配置文件
│  .gitignore                       # Git 忽略文件
│  appBot.css                       # 前端样式表
│  intelllm.py                      # IntelLLM封装
│  logo1.jpg                        # 项目 Logo
│  main.py                          # RAG主程序封装
│  OpenAtom-基于IPEX-LLM生成式AI(AIGC)行业场景应用开发创新赛-法律小助手-zlink.docx # 比赛说明文档
│  requirements.txt                 # Python 依赖库列表
│  runapp.sh                        # Linux 系统运行脚本
│  run_gradio.py                    # Gradio Web 界面入口
│  utils.py                         # 工具函数封装
│
├─custom                            # 自定义模块
│  │  template.py                   # 模板生成逻辑
│  │  transformation.py             # 数据转换功能
│
├─data                              # 数据文件夹
│      刑法.pdf                     # 刑法全文
│      民法典.pdf                   # 民法典全文
│
└─pipeline                          # 数据处理与检索模块
    │  ingestion.py                 # 数据加载与预处理
    │  rag.py                       # 检索增强生成逻辑
```

## 安装与运行

### 安装依赖

1. 在项目根目录运行以下命令，安装所需依赖：
安装依赖需要确保在有效的环境中进行，在runapp.sh打包了一键下载的环境文件：
https://s3.idzcn.com/ipex-llm/ipex-llm-2.1.0b20240410.tar.gz，必须使用这个环境作为虚拟环境。

```bash
pip install -r requirements.txt
```
2. 启动前确保requirements.txt的安装依赖库已经得到了安装，也可以使用一键启动脚本完成依赖环境的安装，详细参考启动服务的shell脚本说明

### 启动服务
#### 方法 1: 使用 Gradio 启动
运行以下命令，启动 Gradio Web 界面：
```bash
python run_gradio.py
```

#### 方法 2: 使用 Shell 脚本启动 (Linux)
运行以下命令：
```bash
sh runapp.sh
```

注意事项：
1. 该脚本会先行下载一个2G左右的压缩文件（参考安装依赖中文件名），展开后作为虚拟环境（创建目录ipex-env），然后激活这个环境后完成requirements.txt的安装；
2. 在intelllm.py文件中会自动下载模型文件（自动创建models目录）：Qwen2.5-1.5B-Instruct，bge-small-zh-v1.5
3.下载完成模型文件后，会在根目录下形成量化模型目录：qwen25-1.5b_int8；（以上的模型如需更改可以在intelllm中修改，目前测试了3B，7B，感觉时间有点长150秒和300秒，达不到实用状态）
4. transformers==4.37.0
5. gradio==4.43.0

## 功能模块说明

### 核心模块
- **main.py**: 封装了rag的核心逻辑。
- **intelllm**：封装了适用于llamaindex的ipex-llm
- **run_gradio.py**：主程序界面。
- **pipeline/ingestion.py**: 处理法律文档的数据加载和预处理。
- **pipeline/rag.py**: 基于 RAG 技术实现检索增强生成。

### 自定义功能
- **custom/template.py**: 定义模板生成逻辑，用于回答结构化输出。
- **custom/transformation.py**: 数据转换逻辑。

### 数据资源
- **data/刑法.pdf** 和 **data/民法典.pdf**: 提供法律条文作为回答依据。

## 相关资料
- **项目 Logo**: `logo1.jpg`
