#!/bin/bash 

# 打印日志
echo "Starting the ipex-llm environment setup..."

# 检查是否已下载 ipex-llm 包
if [ ! -f "ipex-llm-2.1.0b20240410.tar.gz" ]; then
    echo "Downloading ipex-llm package..."
    wget https://s3.idzcn.com/ipex-llm/ipex-llm-2.1.0b20240410.tar.gz
else
    echo "ipex-llm package already downloaded, skipping download."
fi

# 创建解压目录
echo "Creating ipex directory..."
mkdir -p ipex-env

# 检查是否已经解压过
if [ ! -d "ipex-env" ]; then
    echo "Extracting ipex-llm package..."
    tar -zxvf ipex-llm-2.1.0b20240410.tar.gz -C ipex-env/ && rm ipex-llm-2.1.0b20240410.tar.gz
else
    echo "ipex directory already exists, skipping extraction."
fi

# 激活虚拟环境
echo "Activating virtual environment..."
source ipex-env/bin/activate  

# 检查是否已安装依赖
if ! pip show -q -f requirements.txt; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Dependencies already installed, skipping installation."
fi

# 运行 Gradio
echo "Starting Gradio application..."
python run_gradio.py
