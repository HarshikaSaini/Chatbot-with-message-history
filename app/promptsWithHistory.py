from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant",api_key=groq_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all the question in {language}."),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | model
## normally, we would invoke the chain with the input messages and the language, but without message history, the model will not be able to reference the previous messages in the conversation.
response = chain.invoke({"messages":[HumanMessage(content="Hi My name is Harshika. I am a software developer")], "language":"English"})
# print(response.content)

## using message history, we can keep track of the conversation history and use it to generate more relevant responses.
store = {}
def get_session_history(session_id)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(chain,get_session_history,input_messages_key="messages")

config = {"configurable":{"session_id":"chat2"}}
response1 = with_message_history.invoke(
    {"messages":
    [HumanMessage(content="What is my name and what I do?")],
    "language":"English"
    },
    config=config
)

print(response1.content)