import streamlit as st
from langchain_community.utilities import SQLDatabase
import psycopg2
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:postgres@localhost/bigData")

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
        
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: How many different video_id there are in table named brazil?
    SQL Query: SELECT COUNT(DISTINCT video_id) FROM brazil;
    Question: How many video there are for each cannel_title in table india?
    SQL Query: SELECT channel_title, count(*) as num_video FROM india GROUP BY channel_title
    Question: Mi calcoli il rapporto "like" vs "dislike" per ciascun video in Brasile?
    SQL Query: SELECT title, likes, dislikes, CASE WHEN dislikes = 0 THEN NULL ELSE likes / dislikes END AS like_dislike_ratio FROM brazil ORDER BY like_dislike_ratio DESC;
    Question: Mi calcoli la media dei like e delle visualizzazioni per canale in Francia?
    SQL Query: SELECT channel_title, AVG(likes) AS avg_likes, AVG(view_count) AS avg_views FROM france GROUP BY channel_title;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = Ollama(model='llama3')
  llm = ChatGroq(model="llama3-70b-8192", temperature=0)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

def get_response(user_query: str, db: SQLDatabase, content):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response with
    only the data output and without description.
    <SCHEMA>{schema}</SCHEMA>

    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    # content = sql_chain.invoke({"question": user_query}).content
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
        schema=lambda _: db.get_table_info(),
        response=lambda _: db.run(content),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        # "chat_history": chat_history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

st.set_page_config(page_title="Query Augmented Generation", page_icon="speech_balloon:", layout="centered")

st.title("Query Augmented Generation")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        chain = get_sql_chain(db=db)
        content = chain.invoke({"question": user_query})
        
    with st.chat_message("AI"):
        try:
            response = get_response(user_query, db, content)
            st.markdown(response)
        except:
            response = "I' dont understand your question. Please, rewrite it again."
            st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))