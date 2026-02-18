from langchain_core.messages import SystemMessage,trim_messages,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages.utils import count_tokens_approximately
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant",api_key=groq_api_key)
### Example of using trim_messages to trim a list of messages to fit within a token limit. 
# The strategy used here is "last", which means it will keep the last messages in the list until the token limit is reached. 
# The token counter used here is count_tokens_approximately, which gives an approximate token count for the messages.
# The include_system parameter is set to True, which means it will include the system messages in the token count.
#  The allow_partial parameter is set to False, which means it will not include a message if it exceeds the token limit, even if it's the last message in the list.
trimmer = trim_messages(
    max_tokens=50,
    strategy="last",
    token_counter=count_tokens_approximately, ## Either use count_tokens_approximately or model or install transformers and use the model's(fallback model - chtgpt-2) tokenizer to get an exact token count. 
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages=[
    SystemMessage(content="You are a good assistant"),
    HumanMessage(content="What is your name?"),
    AIMessage(content="My name is Groq."),
    HumanMessage(content="My favorite color is blue."),
    AIMessage(content="Nice! BLUE IS A GREAT COLOR."),
    HumanMessage(content="My favorite food is pizza"),
    AIMessage(content="That's a great choice! Pizza is delicious."),
]
## 1. we trimmed the messages to fit within the token limit and then invoked the model with the trimmed messages.
trimmer.invoke(messages)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all the question in {language}."),
    MessagesPlaceholder(variable_name="messages")
])

## 2. we can also use the trimmer as part of a chain to automatically trim the messages before invoking the model. 
# In this example, we are using the trimmer as a RunnablePassthrough to trim the messages before passing them to the prompt and then to the model. 
# This way, we don't have to manually trim the messages every time we want to invoke the model, it will be done automatically as part of the chain.
chain = (RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer )
         | prompt | model)

## 3. we invoke the chain with the input messages and the language, and the trimmer will automatically trim the messages to fit within the token limit before passing them to the prompt and then to the model.
response = chain.invoke({
    "messages":messages + [HumanMessage(content="What is my favorite color?")],
    "language":"English"
})

print(response.content)

## 4. WRAP UP - we can also use the trimmer as part of a RunnableWithMessageHistory to automatically trim the messages in the message history before invoking the model.
store = {}
def get_session_history(session_id)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
    
)
config = {"configurable":{"session_id":"chat1"}}