import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

search = DuckDuckGoSearchRun(name="search")

st.title("agent tool demo")

#sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your groq key",type="password")




if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"hi iam a chat bot  how can i help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key,model = "Llama3-8b-8192",streaming=True)
    tools = [wiki,arxiv,search]

    search_agent = initialize_agent(llm=llm,tools=tools,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)