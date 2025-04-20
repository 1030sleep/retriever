from retriever.config import Config,llamaindexConfig
from llama_index.core import Document
import hashlib

config=Config(chromadb_path="./chromadb",
              docstore_path="./docstore",
              pre_process=False,
              chunk_size=512,
              chunk_overlap=20,
              use_BM25=False,
              model_name="BAAI/bge-small-en-v1.5")

#添加document
texts=["bread for breakfast",
       "hamburger for lunch"]
documents=[]
for text in texts:
    documents.append(Document(text=text,metadata={}))
    # metadata(dict)可以为每个结点添加一些额外的自定义数据,所以这一步没有封装

llamaconfig=llamaindexConfig(config=config,documents=documents)
llamaconfig.build()
retriever=llamaconfig.get_retriever(topk=3)

query1="what for dinner"
nodes=retriever.retrieve(query1)
print("第一次检索：", nodes[0].text)

# 如果想要插入新内容
texts=["beef for dinner",]
documents=[]
for text in texts:
    documents.append(Document(text=text,metadata={}))
llamaconfig.insert(documents=documents)

retriever=llamaconfig.get_retriever(topk=3)
query="what for dinner"
nodes=retriever.retrieve(query)
print("第二次检索：", nodes[0].text)

# 当然只封装了一些最简单的功能,更复杂的功能可以自行参考llamaindex,下面这个函数可以得到其中最核心的组件
docstore,index=llamaconfig.get_docstore_and_index()



