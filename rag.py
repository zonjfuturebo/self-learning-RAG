# from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Tongyi
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import torch
import time


start_time = time.time()
model_name = "bce-embedding-base-v1"

# model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": "True"}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

loader = UnstructuredHTMLLoader("./book_utf8-top500.txt")
docs = loader.load()
print('文档加载完成...')
# split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
split_text = text_splitter.split_documents(docs)
print("文档切分完成...")
# vectordb = Chroma.from_documents(documents=split_text, embedding=hf, persist_directory="./1_db")
vectordb = Chroma(persist_directory="./1_db", embedding_function=hf)
retriever = vectordb.as_retriever()
print('indexing done...')

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)
llm = Tongyi(
    model="qwen-turbo",
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "墨大夫是谁"})
print(response["answer"])

