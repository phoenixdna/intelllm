import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings

#from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data, read_htmldata
#from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval

from intelllm import Intelllm
import logging
import sys
import os

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import nest_asyncio
#import asyncio

#nest_asyncio.apply()

async def main(query_str):
    

    
    config = dotenv_values(".env")
    print("Initializing...")
    from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    intel_llm = Intelllm()

    Settings.embed_model = intel_llm.embed_model
    Settings.llm = intel_llm.llm
    Settings.callback_manager = callback_manager
    
    print("intel llm alreay setup...")
    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "law"
    )
    #nodes = read_htmldata("datah")
    data = read_data("data")
    pipeline = build_pipeline(intel_llm.llm, intel_llm.embed_model, vector_store=vector_store)

    cashe_directory = './pipeline_storage'
    if collection_info.points_count == 0:

        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "law",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        if os.path.exists(cashe_directory) and os.path.isdir(cashe_directory):
            pipeline.load(cashe_directory)

        nodes = list(await pipeline.arun(documents=data, show_progress=True, num_workers=1))
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "law",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))
        #print(len(nodes))
    # save
    pipeline.persist(cashe_directory)

    vector_retriever = QdrantRetriever(vector_store, intel_llm.embed_model, similarity_top_k=5)


    # 生成答案
    print("Start generating answers...")


    print("问题：", query_str)
    result = await generation_with_knowledge_retrieval(
        query_str, vector_retriever, intel_llm.llm, debug=True
    )
    print("答复：", result)
    return str(result)



if __name__ == "__main__":
    asyncio.run(main("小明在保护女友避免被强奸的过程中失手杀死了小王，是否犯法？"))
