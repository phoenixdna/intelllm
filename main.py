import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval2 ,generation_with_knowledge_retrieval1

from intelllm import Intelllm
import logging
import sys
import os
import time

class Main:

    def __init__(self):
        self.config = dotenv_values(".env")
        print("Initializing...")
        self.intel_llm = Intelllm()
        Settings.embed_model = self.intel_llm.embed_model
        Settings.llm = self.intel_llm.llm
        self.vector_retriever = None
        self.reranker = None
        print("intel llm setup done!")

    async def setup(self):
        print("start to setup vector db...")
        if not self.vector_retriever:
            await self.init_vector_db()
        #if not self.reranker:
        #    self.reranker = self.intel_llm.reranker
        print("vector db done!")

    async def init_vector_db(self):
        # 初始化 数据ingestion pipeline 和 vector store
        self.client, self.vector_store = await build_vector_store(self.config, reindex=False)

        collection_info = await self.client.get_collection(self.config["COLLECTION_NAME"] or "law")
        data = read_data("data")
        pipeline = build_pipeline(self.intel_llm.llm, self.intel_llm.embed_model, vector_store=self.vector_store)

        cashe_directory = './pipeline_storage'
        if collection_info.points_count == 0:
            await self.client.update_collection(
                collection_name=self.config["COLLECTION_NAME"] or "law",
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
            )
            if os.path.exists(cashe_directory) and os.path.isdir(cashe_directory):
                pipeline.load(cashe_directory)

            nodes = list(await pipeline.arun(documents=data, show_progress=True, num_workers=1))
            await self.client.update_collection(
                collection_name=self.config["COLLECTION_NAME"] or "law",
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )
            print(len(data))
        pipeline.persist(cashe_directory)

        self.vector_retriever = QdrantRetriever(self.vector_store, self.intel_llm.embed_model, similarity_top_k=20)

    async def myrag(self, query_str):
        print("Start generating rags...")
        print("问题：", query_str)
        st = time.time()
        self.nodes_with_score = await generation_with_knowledge_retrieval1(
            query_str, self.vector_retriever, self.reranker, debug=True
        )
        time.sleep(1)
        print("检索长度：", len(self.nodes_with_score))
        end = time.time()
        print(f'RAG检索时间: {end - st:.2f} s')
        return self.myformat(self.nodes_with_score)

    def myformat(self, refer_list):
        formatted_str = ""
        for idx, node_with_score in enumerate(refer_list[:5], start=1):
            node = node_with_score.node
            score = node_with_score.score
            title = node.metadata.get('document_title', 'Untitled')
            text = node.text
            formatted_str += f"{idx}. Title: {title}\nText: {text}\nScore: {score}\n\n"
        print("formatted_str：\n\n", formatted_str)
        return formatted_str

    async def main(self, query_str):
        print("Start generating answers...")
        print("问题：", query_str)
        st = time.time()
        result = await generation_with_knowledge_retrieval2(
            query_str, self.nodes_with_score, self.intel_llm.llm, self.reranker, debug=True
        )
        print("答复：", result)
        end = time.time()
        print(f'Inference time: {end - st:.2f} s')
        return str(result)

if __name__ == "__main__":
    mymain = Main()
    # 运行异步的初始化操作
    asyncio.run(mymain.setup())
    asyncio.run(mymain.myrag("小明在保护女友避免被强奸的过程中失手杀死了小王，是否犯法？"))
    asyncio.run(mymain.main("小明在保护女友避免被强奸的过程中失手杀死了小王，是否犯法？"))
