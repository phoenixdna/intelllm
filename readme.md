# 法律小助手

## 项目简介
本项目基于 IPEX-LLM 技术开发，旨在为用户提供法律相关问题的智能解答服务。通过解析用户输入问题，引用相关法律条文进行回答，适用于法律咨询等场景。

## 软件运行环境要求
- **CPU**: 6核及以上
- **内存**: 16GB及以上
- **Python 版本**: 3.9+
- **Torch版本**：2.2.2+

## 项目目录结构
```
│  .env                             # 环境变量配置文件
│  .gitignore                       # Git 忽略文件
│  appBot.css                       # 前端样式表
│  intelllm.py                      # 智能模块核心逻辑
│  logo1.jpg                        # 项目 Logo
│  main.py                          # 主入口脚本
│  OpenAtom-基于IPEX-LLM生成式AI(AIGC)行业场景应用开发创新赛-法律小助手-zlink.docx # 比赛说明文档
│  requirements.txt                 # Python 依赖库列表
│  runapp.sh                        # Linux 系统运行脚本
│  run_gradio.py                    # Gradio Web 界面入口
│  utils.py                         # 工具函数
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

注意事项：1. 该脚本会先行下载一个2G左右的文件，展开后作为虚拟环境，要确保在这个环境下完成requirements.txt的安装；
2. 需要使用transformers==4.37.0
3. 需要使用gradio==4.43.0

## 功能模块说明

### 核心模块
- **main.py**: 封装了rag的核心接口。
- **intelllm**：封装了适用于llamaindex和ipex的llm
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
- **比赛说明文档**: `OpenAtom-基于IPEX-LLM生成式AI(AIGC)行业场景应用开发创新
