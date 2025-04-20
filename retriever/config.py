from dataclasses import dataclass

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import Document
import chromadb
import hashlib

@dataclass
class Config:
    chromadb_path: str # 向量库存储路径
    docstore_path: str # 预处理文本存储路径

    pre_process: bool # 是否预处理
    chunk_size: int # 切片大小
    chunk_overlap:int # 每个切片允许重叠的范围

    use_BM25: bool # 是否使用BM25
    model_name: str # hf_model_name or file path
    retriever_weights: list[float]=None # 召回权重

class llamaindexConfig():
    def __init__(self,
                 config: Config=None,
                 documents:list[str]=None) :
        self.config=config
        self.documents=documents
        self.embed_model=HuggingFaceEmbedding(model_name=self.config.model_name)
        self.docstore=None
        self.index=None
        # 定义 NodeParser（按固定字符长度分割，设置足够大的 chunk_size 避免分割）
        self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.config.chunk_size,  # 假设段落不超过512字符
                chunk_overlap=self.config.chunk_overlap,
            )
        self.client=None
        # 由于没有用到生成功能，先禁用llm（否则默认调用openai gpt3.5
        Settings.llm=None
        
    
    def build(self):
        # 先对文本进行预处理

        if self.config.pre_process and self.documents!= None:
            
            nodes = self.node_parser.get_nodes_from_documents(self.documents)
            print(f"Created {len(nodes)} nodes.")
        
            docstore=SimpleDocumentStore()
            docstore.add_documents(nodes)
            docstore.persist(self.config.docstore_path)
        
        elif not self.config.pre_process:
            #已经预处理过
            docstore=SimpleDocumentStore.from_persist_path(self.config.docstore_path)
        
        # 通过文本构建向量存储
        if not self.config.pre_process:
            # 持久化存储
            db = chromadb.PersistentClient(path=self.config.chromadb_path)
            self.client=db
            chroma_collection = db.get_or_create_collection("dense_vectors")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store,embed_model=self.embed_model)
        
        else:
            # 持久化存储
            db = chromadb.PersistentClient(path=self.config.chromadb_path)
            self.client=db
            chroma_collection = db.get_or_create_collection("dense_vectors")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
            storage_context = StorageContext.from_defaults(
                docstore=docstore, vector_store=vector_store
            )
            index = VectorStoreIndex(nodes=nodes, storage_context=storage_context,embed_model=self.embed_model,show_progress=True)
            
        self.docstore=docstore
        self.index=index
        
        return docstore,index
    
    def get_retriever(self , topk=1):
        if self.config.use_BM25:
            assert self.config.retriever_weights !=None, "权重未知"
            print("enable BM25")
            retriever = QueryFusionRetriever(
                [
                    self.index.as_retriever(similarity_top_k=topk),
                    BM25Retriever.from_defaults(
                        docstore=self.docstore, similarity_top_k=topk
                    ),
                ],
                retriever_weights=self.config.retriever_weights,
                num_queries=1,
                mode="dist_based_score",
                use_async=False,
            )
        else:
            print("disable BM25")
            retriever=self.index.as_retriever(similarity_top_k=topk)
        
        print("retriever built")
        
        return retriever
    
    def insert(self,documents:list[Document]):
        new_node_num=0
        for document in documents:
            for doc_id in self.docstore.docs:
                existing_doc = self.docstore.get_document(doc_id)
                existing_hash = self.get_content_hash(existing_doc.text)
                if existing_hash == self.get_content_hash(document.text):
                    exists = True
                    print("重复nodes")
                    break
            new_node=self.node_parser.get_nodes_from_documents(documents=[document])
            new_node_num+=1
            self.docstore.add_documents([document])
            self.docstore.persist(self.config.docstore_path)
            self.index.insert_nodes(new_node)
        print(f"Created {new_node_num} new nodes.")
                
    
    def get_docstore_and_index(self):
        return self.docstore,self.index
    
    def get_content_hash(self,text):
        return hashlib.md5(text.encode()).hexdigest()





    
