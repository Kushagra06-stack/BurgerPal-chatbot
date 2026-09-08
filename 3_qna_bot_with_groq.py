from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_keenable import KeenableSearch
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

model = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = KeenableSearch()

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []



memory = MemorySaver()

@tool
def web_search(query: str) -> str:
    """Search the web and return only a limited amount of relevant information."""
    
    result = search.invoke(query)
    
    # Convert search result to text
    text = str(result)
    
    # Keep only first 6000 characters
    return text[:6000]

# Create agent
agent = create_agent(
    model=model,
    tools=[web_search],
    checkpointer=st.session_state.memory,
    system_prompt="You are a web search agent. Search the web and answer the user's question using the available search results."
)

print(st.session_state.memory)

##Building Web Interface
st.subheader("🤖BurgerPal🍔 - BurgerAI")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user", "content":(query)})


    response = agent.stream(
        {"messages":[{"role":"user", "content":(query)}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages"
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message+chunk[0].content
            space.write(message)

        st.session_state.history.append({"role":"ai", "content":(message)})
