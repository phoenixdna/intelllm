# 设置OpenMP线程数为8
import os
import time
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from typing import Any, List, Optional

# 从llama_index库导入所需的类
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.vector_stores.chroma import ChromaVectorStore
#from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
#import chromadb
from ipex_llm.llamaindex.llms import IpexLLM

class Config:
    """配置类,存储所有需要的参数"""
    model_path = "qwen2chat_7b_int4"
    tokenizer_path = "qwen2chat_7b_int4"
#    question = "How does Llama 2 perform compared to other open-source models?"
    data_path = "./data/llamatiny.pdf"
    persist_dir = "./chroma_db"
    embedding_model_path = "qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
    max_new_tokens = 512
'''
def load_vector_database(persist_dir: str) -> ChromaVectorStore:
    """
    加载或创建向量数据库
    
    Args:
        persist_dir (str): 持久化目录路径
    
    Returns:
        ChromaVectorStore: 向量存储对象
    """
    if os.path.exists(persist_dir):
        print(f"正在加载现有的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection("llama2_paper")
    else:
        print(f"创建新的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.create_collection("llama2_paper")
    print(f"Vector store loaded with {chroma_collection.count()} documents")
    return ChromaVectorStore(chroma_collection=chroma_collection)

def load_data(data_path: str) -> List[TextNode]:
    """
    加载并处理PDF数据
    
    Args:
        data_path (str): PDF文件路径
    
    Returns:
        List[TextNode]: 处理后的文本节点列表
    """
    loader = PyMuPDFReader()
    documents = loader.load(file_path=data_path)

    text_parser = SentenceSplitter(chunk_size=384)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes

class VectorDBRetriever(BaseRetriever):
    """向量数据库检索器"""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        检索相关文档
        
        Args:
            query_bundle (QueryBundle): 查询包
        
        Returns:
            List[NodeWithScore]: 检索到的文档节点及其相关性得分
        """
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores
'''
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
        context_window=2048,
        max_new_tokens=config.max_new_tokens,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        model_kwargs={},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="cpu",
    )


class Intelllm:
    def __init__(self) -> None:
        self.config = Config()
        self.embed_model = HuggingFaceEmbedding(model_name=self.config.embedding_model_path)
        self.llm = setup_llm(self.config)


