from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor


#import nest_asyncio
#nest_asyncio.apply()


def read_data(path: str = "data") -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        #encoding="gbk",
        required_exts=[
            ".pdf",
        ],
    )
    return reader.load_data()



#from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.schema import BaseNode
def read_file_htmldata(filepath: str) -> list[BaseNode]:
    reader = myFlatReader()
    demo_file = reader.load_data(Path(filepath))
    node_parser = UnstructuredElementNodeParser()
    raw_nodes = node_parser.get_nodes_from_documents(demo_file)
    base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(raw_nodes)
    for node in base_nodes:
        node.metadata['file_path'] = filepath
    return base_nodes

import os
from pathlib import Path
from typing import List
def read_htmldata(path:str) -> list[BaseNode]:
    filepaths = find_files_with_extension(path)
    all_nodes = []
    for filepath in filepaths:
        nodes = read_file_htmldata(filepath)
        all_nodes.extend(nodes)
    return all_nodes



def find_files_with_extension(directory: str, extension: str = 'html') -> List[str]:
    """
    在指定目录及其子目录下查找具有指定扩展名的文件，并返回它们的相对路径列表。

    :param directory: 要搜索的目录路径
    :param extension: 要搜索的文件扩展名，默认为 'html'
    :return: 带有指定扩展名的文件相对路径列表
    """
    directory_path = Path(directory)
    matching_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(f'.{extension}'):
                file_path = Path(root) / file
                # 构建相对路径，并在前面添加目录名
                relative_path = directory_path / file_path.relative_to(directory_path)
                matching_files.append(str(relative_path))
    return matching_files





def build_pipeline(
        llm: LLM,
        embed_model: BaseEmbedding,
        template: str = None,
        vector_store: BasePydanticVectorStore = None,
) -> IngestionPipeline:
    #dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX)
    transformation = [
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=7, metadata_mode=MetadataMode.EMBED),
        #QuestionsAnsweredExtractor(
        #    questions=3, metadata_mode=MetadataMode.EMBED
        #),
        #SummaryExtractor(
        #    #llm=dashscope_llm,
        #    metadata_mode=MetadataMode.EMBED,
        #    prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        #),
        embed_model,
    ]

    return IngestionPipeline(transformations=transformation, vector_store=vector_store)


async def build_vector_store(
        config: dict, reindex: bool = False
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    client = AsyncQdrantClient(
        # url=config["QDRANT_URL"],
        location=":memory:"
    )
    if reindex:
        try:
            await client.delete_collection(config["COLLECTION_NAME"] or "law")
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")

    try:
        await client.create_collection(
            collection_name=config["COLLECTION_NAME"] or "law",
            vectors_config=models.VectorParams(
                size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
            ),
        )
    except UnexpectedResponse:
        print("Collection already exists")
    return client, QdrantVectorStore(
        aclient=client,
        collection_name=config["COLLECTION_NAME"] or "aiops24",
        parallel=4,
        batch_size=32,
    )


from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class myFlatReader(BaseReader):
    """Flat reader.

    Extract raw text from a file and save the file type in the metadata
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file into string."""
        with open(file, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        metadata = {"filename": file.name, "extension": file.suffix}
        if extra_info:
            metadata = {**metadata, **extra_info}

        return [Document(text=content, metadata=metadata)]