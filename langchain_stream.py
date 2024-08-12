from dashscope import Generation

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time


# 记录时间
start_time = time.time()
time_list = []
messages = []
# 加载模型
model_name = r'stella_en_1.5B_v5'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

db = Chroma(persist_directory='./db', embedding_function=hf)  # 加载数据

while True:
    message = input("User>> ")

    similarDocs = db.similarity_search(message, k=5)
    summary_prompt = "".join([doc.page_content for doc in similarDocs])

    send_message = f"下面的信息{summary_prompt}是否有这个问题{message}有关，如果你觉得无关请告诉我无法根据提供的上下文回答'{message}'这个问题，简要回答即可，否则请根据{summary_prompt}对{message}的问题进行回答 "
    messages.append({'role': 'user', 'content': send_message})
    whole_message = ''
    # 切换模型
    responses = Generation.call(
        model="qwen-turbo",
        messages=messages,
        result_format='message',
        stream=True,
        incremental_output=True
        )
    print('system>> ', end='')
    for response in responses:
        whole_message += response.output.choices[0].message.content
        print(response.output.choices[0].message.content, end='')
    print()
    messages.append({'role': 'assistant', 'content': whole_message})
