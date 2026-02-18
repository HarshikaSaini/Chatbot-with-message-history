from langchain_core.messages import SystemMessage,trim_messages,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages.utils import count_tokens_approximately
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

response = trimmer.invoke(messages)
print(response)