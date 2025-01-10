from typing import List
import qdrant_client

from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse

from custom.template import QA_TEMPLATE
###新增import

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.schema import (
    BaseNode,
    Document,
    MetadataMode,
    NodeRelationship,
    TransformComponent,
)


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 10,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores


async def generation_with_knowledge_retrieval1(
    query_str: str,
    retriever: BaseRetriever,
    reranker: BaseNodePostprocessor = None,
    debug: bool = False,

) :
    query_bundle = QueryBundle(query_str=query_str)
    #from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
    #dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX_1201)

    node_with_scores = await retriever.aretrieve(query_bundle)
    if debug:
        print("retrieved nodes Number:", len(node_with_scores))
        # print(f"retrieved:\n{node_with_scores}\n------")
        # 打印每个node的编号、text和score
        # for i, node in enumerate(node_with_scores, start=1):
        #    print(f"{i}. {node.text} \n: {node.score}\n")
        for node in node_with_scores:
            print(node)
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
            print("."*50+"after reranking"+"."*50)
            for i, node in enumerate(node_with_scores, start=1):
               print(f"{i}. {node.text} \n: {node.score}\n")
    return node_with_scores



async def generation_with_knowledge_retrieval2(
        query_str: str,
        node_with_scores,
        #retriever: BaseRetriever,
        llm: LLM,
        reranker: BaseNodePostprocessor = None,
        qa_template: str = QA_TEMPLATE,
        debug: bool = False,

) -> CompletionResponse:


    query_bundle = QueryBundle(query_str=query_str)

    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
            print("after reranking.....")
            for i, node in enumerate(node_with_scores, start=1):
               print(f"{i}. {node.text} \n: {node.score}\n")

    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores[:3]]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )

    ret = await llm.acomplete(fmt_qa_prompt)
    # print("精炼前回答：", ret.text)

    '''
        from llama_index.core.response_synthesizers import Refine

        refine_prompt_tmpl = (
            "原始查询如下：{query_str}\n"
            "我们已提供现有答案：{existing_answer}\n"
            "我们有机会通过以下更多上下文来改进现有答案"
            "下面提供更多的上下文信息。\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "（仅在{existing_answer}不确定时）鉴于新的上下文，改进原始答案以更好地回答查询。"
            "如果上下文无用，则返回{existing_answer}。\n"
            "改进后的答案："
        )
        refine_prompt = PromptTemplate(refine_prompt_tmpl)
        texts = []
        for item in node_with_scores:
            texts.append(item.get_text())

        summarizer = Refine(llm=dashscope_llm, refine_template=refine_prompt, verbose=True)
        response = summarizer.get_response(query_str, texts)
        print("精炼以后回答:", response)
        ret.text = response
    '''
    return ret
