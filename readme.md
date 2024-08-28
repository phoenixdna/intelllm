首先感谢主办方和intel，该项目主要构建了一个基于Qwen2 7B CPU量化INT4/INT8模型的法律智能助手，专注于刑法和民法领域，能够准确解读和应用相关法律条文，为用户提供即时、经济高效的法律咨询和案件分析服务。2024-08-28

使用的主要软件包：
- IPEX：ipex-llm（Intel® Extension for PyTorch Large Language Models）：是一个基于 PyTorch 的库，主旨在 Intel CPU 和 GPU 上高效运行大型语言模型（LLM）。该库特别适用于本地 PC 上的集成显卡（iGPU）和独立显卡（如 Arc、Flex 和 Max），能够以极低的延迟进行推理和微调。
- LLamaIndex：LlamaIndex 是一个用于构建大型语言模型（LLM）数据应用的框架。它支持多种语言模型的集成，使得开发者能够更高效地处理和利用大规模数据，特别是在信息检索、问答系统和知识管理等应用场景中。

主要的硬件依赖：
IntelCPU：
- 比如i5-12490；32G内存
- 阿里云的g8i-6xlargeCPU：Intel(R) Xeon(R) Platinum 8575C @ 3.00GHz，核心数24；内存96G


相关步骤如下：

1. 需要先安装requirements.txt所有依赖；
2. 然后进行模型下载，downlaod.py
3. 进行emb模型下载，downlaod2.py
4. 进行模型量化，并完成本地存储：quantization.py（建议选择int4）
5. 运行run_graido.py进行法律援助问答

说下结论：
当前使用了比赛要求的g8i-6xlarge的情况：
- 对于rag查询基本都是10秒级别（8-9秒不等）
- 对于模型推理如果是int4，在40-60秒时间不等；
- 对于模型推理如果是int8，在80-100秒时间不等；

具体可以参考我PR里面的output_performance.txt（终端输出）