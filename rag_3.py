from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Tongyi
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 加载大模型，接入API
llm = Tongyi(
    model='qwen-turbo',
    tempture=0.8
)

# 加载嵌入模型
model_name = r'bce-embedding-base-v1'
model_kwargs = {'device': 'cuda'}
encode_kwargs = {"normalize_embeddings": "True"}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# 加载向量数据库
vectordb = Chroma(persist_directory="./1_db", embedding_function=hf)
retriever = vectordb.as_retriever()
print('indexing done...')

# 设置提示词模板
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Question: {input}")
])

# History aware retriever
contextualize_q_system_prompt = """Using the chat history and the user's question, create a standalone question that 
can be understood without the chat history. formulate the question if necessary; otherwise, return it as is. 
**Important!** Do not directly answer the user's question.
"""
history_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Question: {input}")
])
history_retriever = create_history_aware_retriever(
    llm,  # LanguageModelLike
    retriever,  # VectorStoreRetriever
    history_prompt  # ChatPromptTemplate
)

# RAG Chain
basic_qa_chain = create_stuff_documents_chain(
    llm,
    prompt,
)
# basic_qa_chain = prompt | llm
qa_chain = create_retrieval_chain(
    history_retriever,
    basic_qa_chain
)
# qa_chain = history_retriever | basic_qa_chain

# 接入历史记录
DEFAULT_MAX_MESSAGES = 10
store = {}  # 存储历史对话记录


class LimitedChatMessageHistory(ChatMessageHistory):  # ChatMessageHistory继承了BaseChatMessageHistory类
    # messages: List[BaseMessage] = Field(default_factory=list)
    max_messages: int = DEFAULT_MAX_MESSAGES

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES):
        super().__init__()
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, added_message):
        super().add_message(added_message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]  # 从后往前切10条，即仅保存最近的10条历史信息

    def get_messages(self):
        return self.messages


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = LimitedChatMessageHistory()
    return store[session_id]


runnable_with_history = RunnableWithMessageHistory(
    runnable=qa_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

session_id = "1"
response = runnable_with_history.invoke(
    {"input": "韩立是谁？"},
    config={"configurable": {"session_id": session_id}}
)
print(response)
print("\n\n"+response["answer"])

response = runnable_with_history.invoke(
    {"input": "我的名字叫bob"},
    config={"configurable": {"session_id": session_id}}
)
print(response)
print("\n\n"+response["answer"])

response = runnable_with_history.invoke(
    {"input": "回答一下，bob和韩立都是谁？"},
    config={"configurable": {"session_id": session_id}}
)
print(response)
print("\n\n"+response["answer"])
