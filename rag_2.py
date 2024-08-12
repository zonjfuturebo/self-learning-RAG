from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.llms import Tongyi
# 直接使用Tongyi报错，或考虑使用提示模板
# from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 设置最大历史对话记录数
DEFAULT_MAX_MESSAGES = 10

llm = Tongyi(
    model="qwen-turbo",
    temperature=0.7,
)


# 新建一个可以设置最大历史对话记录数的类LimitedChatMessageHistory
class LimitedChatMessageHistory(ChatMessageHistory):
    # 最大对话记录数
    max_messages: int = DEFAULT_MAX_MESSAGES

    def __init__(self, max_messages=DEFAULT_MAX_MESSAGES):
        super().__init__()
        self.max_messages = max_messages

    def add_message(self, message):
        super().add_message(message)
        # 调整历史记录
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        return self.messages


# 存储历史对话记录
store = {}


# 根据Session ID获取历史对话记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = LimitedChatMessageHistory()
    return store[session_id]


# 使用prompt提示模板编写
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个智能助手，请根据用户输入回答问题。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt_chain = prompt | llm

prompt_runnable = RunnableWithMessageHistory(
    prompt_chain,
    get_session_history
)

prompt_config = {"configurable": {"session_id": "prompt"}}
response_1 = prompt_runnable.invoke(
    {"message": [HumanMessage(content="你好，我叫bob!")]},
    config=prompt_config,
)
print(response_1)
runnable_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

answer = runnable_with_history.invoke(
    "你好，我叫bob!",
    config={"configurable": {"session_id": "1"}},
)
print(answer)
# answer = runnable_with_history.invoke(
#     {"message": [HumanMessage(content="我叫什么名字？")]},
#     config={"configurable": {"session_id": "1"}},
# )
# print(answer)
# answer = runnable_with_history.invoke(
#     [HumanMessage(content="我喜欢看《凡人修仙传》这本小说，所以，我现在名字改成韩立了。")],
#     config={"configurable": {"session_id": "1"}},
# )
# print(answer)
# answer = runnable_with_history.invoke(
#     [HumanMessage(content="我叫什么名字？")],
#     config={"configurable": {"session_id": "1"}},
# )
# print(answer)
# answer = runnable_with_history.invoke(
#     [HumanMessage(content="我以前叫什么？")],
#     config={"configurable": {"session_id": "1"}},
# )
# print(answer)
