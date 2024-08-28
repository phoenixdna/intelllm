import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings

#from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data, read_htmldata
#from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval2 ,generation_with_knowledge_retrieval1

from intelllm import Intelllm
import logging
import sys
import os

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import nest_asyncio
#import asyncio

#nest_asyncio.apply()

class Main():

    def __init__(self):

        self.config = dotenv_values(".env")
        print("Initializing...")
        from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])

        self.intel_llm = Intelllm()

        Settings.embed_model = self.intel_llm.embed_model
        Settings.llm = self.intel_llm.llm
        Settings.callback_manager = callback_manager

        print("intel llm alreay setup...")

    async def myrag(self,query_str):
        # 初始化 数据ingestion pipeline 和 vector store
        self.client, self.vector_store = await build_vector_store(self.config, reindex=False)

        collection_info = await self.client.get_collection(
            self.config["COLLECTION_NAME"] or "law"
        )
        # nodes = read_htmldata("datah")
        data = read_data("data")
        pipeline = build_pipeline(self.intel_llm.llm, self.intel_llm.embed_model, vector_store=self.vector_store)

        cashe_directory = './pipeline_storage'
        if collection_info.points_count == 0:

            # 暂时停止实时索引
            await self.client.update_collection(
                collection_name=self.config["COLLECTION_NAME"] or "law",
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
            )
            if os.path.exists(cashe_directory) and os.path.isdir(cashe_directory):
                pipeline.load(cashe_directory)

            nodes = list(await pipeline.arun(documents=data, show_progress=True, num_workers=1))
            # 恢复实时索引
            await self.client.update_collection(
                collection_name=self.config["COLLECTION_NAME"] or "law",
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )
            print(len(data))
            # print(len(nodes))
        # save
        pipeline.persist(cashe_directory)

        self.vector_retriever = QdrantRetriever(self.vector_store, self.intel_llm.embed_model, similarity_top_k=5)

        # 生成答案
        print("Start generating rags...")

        print("问题：", query_str)
        self.nodes_with_score = await generation_with_knowledge_retrieval1(
            query_str, self.vector_retriever, debug= True
        )
        print("检索长度：", len(self.nodes_with_score))

        return self.myformat(self.nodes_with_score)

    def myformat(self, refer_list):
        formatted_str = ""
        for idx, node_with_score in enumerate(refer_list, start=1):
            node = node_with_score.node
            score = node_with_score.score
            title = node.metadata.get('document_title', 'Untitled')
            text = node.text

            # 按照指定格式生成字符串
            formatted_str += f"{idx}. Title: {title}\nText: {text}\nScore: {score}\n\n"

        print("formatted_str：\n\n", formatted_str)
        return formatted_str

    async def main(self,query_str):

        # 生成答案
        print("Start generating answers...")


        print("问题：", query_str)
        result = await generation_with_knowledge_retrieval2(
            query_str, self.nodes_with_score, self.intel_llm.llm, debug=True
        )
        print("答复：", result)

        return str(result)



if __name__ == "__main__":
    mymain= Main()
    asyncio.run(mymain.myrag("小明在保护女友避免被强奸的过程中失手杀死了小王，是否犯法？"))
    asyncio.run(mymain.main("小明在保护女友避免被强奸的过程中失手杀死了小王，是否犯法？"))
