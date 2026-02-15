from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant",api_key=groq_api_key)
result = model.invoke([
    HumanMessage(content="Hi, My name is Harshika. I am a software developer"),
    AIMessage(content="Hi Harshika! Nice to meet you. How can I assist you today?"),
    HumanMessage(content="What is my name and what I do?")
])

### Message History
'''
We can use Message History to wrap our model and make it stateful. 
This allows us to keep track of the conversation history (input and output) and use it to generate more relevant responses.
Future interactions with the model will be able to reference the previous messages in the conversation,
and pass them into chain as a part if input.

'''
store = {}
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

## using the RunnableWithMessageHistory wrapper, we can now invoke the model with a session_id, and it will automatically keep track of the conversation history for that session.
with_message_history = RunnableWithMessageHistory(model,get_session_history)

config = {"configurable":{"session_id":"chat1"}}
history =  with_message_history.invoke(
    [HumanMessage(content="Hi, My name is Harshika. I am a software developer")],
    config=config
)
## IF WE INVOKE THE MODEL AGAIN WITH THE SAME SESSION ID, 
# IT WILL BE ABLE TO REFERENCE THE PREVIOUS MESSAGES IN THE CONVERSATION HISTORY. 
# BUT IF WE USE A DIFFERENT SESSION ID, IT WILL START A NEW CONVERSATION HISTORY.
config1 = {"configurable":{"session_id":"chat2"}}
history1 =  with_message_history.invoke(
    [HumanMessage(content="What is my name and what I do?")],
    config=config1 
)

print(history1.content)
