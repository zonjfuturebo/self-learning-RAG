from langchain_community.chat_models import ChatTongyi
from langchain_community.llms import Tongyi
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

# 全局变量，使用内存存储历史记录
store = {}
MAX_HISTORY = 10
# 加载llm
llm_chattongyi = ChatTongyi(model_name="qwen-turbo")
llm_tongyi = Tongyi(model_name="qwen-turbo")
llm = ChatOllama(model='qwen2')

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个金融专家，请根据我的提问，提供准确的历史信息。"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "Query: {query}")
])
# 创建初始链
chain = prompt | llm


# 修改chain的聊天历史
class HistoryWithTwoKeys(ChatMessageHistory):
    max_messages: int = MAX_HISTORY

    def __init__(self, max_messages: int = MAX_HISTORY):
        super().__init__()
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, message):
        super().add_message(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        return self.messages


# 定义获取session_id
def get_history_id(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = HistoryWithTwoKeys()
    return store[(user_id, conversation_id)]


# 实例化
history_runnable = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_history_id,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        )
    ]
)
response = history_runnable.invoke(
    {"query": "我叫bob，请记住我的国家：委内瑞拉"},
    config={"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}
)
print(response)
response = history_runnable.invoke(
    {"query": "我是谁，我的国家是哪里"},
    config={"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}
)
print(response)
response = history_runnable.invoke(
    {"query": "我是谁，我的国家是哪里"},
    config={"configurable": {"user_id": "user1", "conversation_id": "conversation2"}}
)
print(response)
response = history_runnable.invoke(
    {"query": "我是Alice，我来自伦敦"},
    config={"configurable": {"user_id": "user2", "conversation_id": "conversation1"}}
)
print(response)
response = history_runnable.invoke(
    {"query": "我是谁"},
    config={"configurable": {"user_id": "user2", "conversation_id": "conversation1"}}
)
print(response)
