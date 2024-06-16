from groq import APIStatusError
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
from valutation import append_to_json


load_dotenv()

URI = "postgresql+psycopg2://postgres:postgres@localhost/youTubeDataset"
db = SQLDatabase.from_uri(URI)

# file_path = "./test/complex_valutation.json"

# def create_views_for_query(uri,)

def get_database_info(db):
  template = """
    You are information assistant. 
    Based on the table schema below, write an answer that describe the attributes and the name of each table in database.
    
    <SCHEMA>{schema}</SCHEMA>
        
    Your turn:
    
    Question: {question}
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )
   

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query for a PostgreSQL database, that would answer the user's question. Take the conversation history into account.
    
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
    Question: Mi trovi i 5 video con più like in united states che sono della categoria Animation, restituendo id del video e id del canale?
    SQL Query: SELECT video_id, channel_id FROM united_states \nWHERE category_id = (SELECT category_id FROM category WHERE title = 'Animation') ORDER BY likes DESC LIMIT 5;
    Question: Trova i video pubblicati nel 2020 in brasile?
    SQL Query: SELECT 'brazil' AS source, title, published_date FROM brazil WHERE published_date BETWEEN '2020-01-01' AND '2020-12-31';
    Question: Trova i video pubblicati nel 2020 in brasile e tra le 17 e le 19?
    SQL Query: SELECT 'brazil' AS source, title, published_date FROM brazil WHERE published_date BETWEEN '2020-01-01' AND '2020-12-31' AND published_time BETWEEN '17:00:00' AND '19:00:00';
    Question: mi trovi la published_data in cui si è registrato il maggior numero di visualizzazioni tra marzo 2021 e agosto 2022 in canada, restituendo data di publicazione e numero di visualizzazioni?
    SQL Query: SELECT published_date, view_count FROM canada WHERE published_date BETWEEN '2021-03-01' AND '2022-08-31' ORDER BY view_count DESC LIMIT 1;
    Question: count the average number of videos for each tables on database.
    SQL Query: SELECT AVG(num_videos) from (SELECT COUNT(*) as num_videos FROM brazil
                UNION ALL SELECT COUNT(*) FROM canada
                UNION ALL SELECT COUNT(*) FROM korea
                UNION ALL SELECT COUNT(*) FROM france
                UNION ALL SELECT COUNT(*) FROM germany
                UNION ALL SELECT COUNT(*) FROM great_britain
                UNION ALL SELECT COUNT(*) FROM india
                UNION ALL SELECT COUNT(*) FROM japan
                UNION ALL SELECT COUNT(*) FROM mexico
                UNION ALL SELECT COUNT(*) FROM russia
                UNION ALL SELECT COUNT(*) FROM united_states)
    Question: compute the number of video in table mexico with comment disabled?
    SQL Query: SELECT COUNT(*) FROM mexico WHERE comments_disabled = TRUE;
    Question: Count the number of videos in the Korea table with the same title as videos in the Japan table.
    SQL Query: SELECT COUNT(*) FROM korea k INNER JOIN japan j ON k.title = j.title;
    

    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = Ollama(model='llama3')
  llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)
  
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
    only the data output and without description or query sql.
    <SCHEMA>{schema}</SCHEMA>

    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.75)
    # content = sql_chain.invoke({"question": user_query}).content
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
        # query=lambda _: content,
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
      AIMessage(
        content="""
        Hello! I'm a SQL assistant.\n
        The dataset represents trending YouTube videos in various countries. It contains separate tables for this 
        country (Brazil, Canada, France, Germany, Great Britain, India, Japan, Korea, and Mexico), each with detailed 
        information on trending videos in that country. Each table includes fields such as video ID, title, 
        publication date, channel ID, channel title, category ID, trending date, tags, view count, likes, dislikes, 
        comment count, thumbnail link, whether comments are disabled, whether ratings are disabled, and video description. 
        There is also a category table that maps category IDs to their titles.\n
        Ask me anything about database."""),
    ]

# st.set_page_config(page_title="Query Augmented Generation", page_icon="speech_balloon:", layout="centered")

st.set_page_config(
    page_title="Query Augmented Generation",
    page_icon="llm.jpg",
    layout="wide",
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("Query Augmented Generation")
# with col2:
#    st.image("llm.jpg")

# st.title("Query Augmented Generation")

with st.expander("Write this sentences if you want more information...", expanded=True):
    st.write(
        """
        - What table and his attributes there are in this database?
        - Count the average number of videos for each tables on database.
        - Count the number of videos on database.
    """
    )

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    # start_time = time.time()

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    if user_query == "stop":
        st.stop()

    elif user_query.startswith("What table and his attributes"):
        with st.chat_message("Human"):
            st.markdown(user_query)
            llm = get_database_info(db=db)
            response = llm.invoke({"question": user_query})
            
        with st.chat_message("AI"):
            st.markdown(response)

    else:
        with st.chat_message("Human"):
            st.markdown(user_query)
            chain = get_sql_chain(db=db)
            content = chain.invoke({"question": user_query})
            print(content)
            
        with st.chat_message("AI"):
            try:
                response = get_response(user_query, db, content)
                print(response)
                st.markdown(response)

            except Exception as exception:
                response = "Request too large."
                st.markdown(response)
                # if exception.__class__ is APIStatusError:
                #     response = "Request is very large. Please add any attributes."
                #     st.markdown(response)
                # else:
                #     response = "I' dont understand your question. Please, rewrite it again."
                #     st.markdown(response)

    # end_time = time.time()
    # execution_time = end_time - start_time
    # append_to_json(user_input=user_query, sql_query=content, system_output=response, execution_time = execution_time, file_path=file_path)
    st.session_state.chat_history.append(AIMessage(content=response))