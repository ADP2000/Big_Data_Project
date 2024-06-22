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
from pyspark.sql import SparkSession
import os


load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
db = SQLDatabase.from_uri(URI)

# file_path = "./times/execution_time_complex.json"

def execute_query_and_get_rows_as_tuple_string(query, spark):
    df = spark.sql(query)
    
    result = df.collect()
    
    columns = df.columns
    
    rows_as_tuples = [tuple(row[col] for col in columns) for row in result]
    
    result_str = str(rows_as_tuples)
    
    return result_str

def create_views_for_query(spark):
  jdbc_url = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"
  properties = {
      "user": DB_USER,
      "password": DB_PASSWORD,
      "driver": "org.postgresql.Driver"
  }

  conn = psycopg2.connect(
      dbname=DB_NAME,
      user=DB_USER,
      password=DB_PASSWORD,
      host=DB_HOST,
      port=DB_PORT
  )
  cursor = conn.cursor()
  cursor.execute("""
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
  """)
  tables = cursor.fetchall()
  conn.close()

  dataframes = {}
  for (table_name,) in tables:
    df = spark.read.jdbc(url=jdbc_url, table=table_name, properties=properties)
    dataframes[table_name] = df

  for table_name, df in dataframes.items():
      df.createOrReplaceTempView(table_name)


def get_database_info(db):
  template = """
    Hello! I'm an information assistant.

    The dataset represents [Brief Description of the Database]. Here is a detailed overview of the tables and their attributes in the database:

    <SCHEMA>{schema}</SCHEMA>

    Feel free to ask me anything about the database.
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
    
    Your turn:
    
    Question: {question}
    SQL Query:
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

def get_response(user_query: str, db: SQLDatabase, content, spark):
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
        # response=lambda _: db.run(content),
        response = lambda _: execute_query_and_get_rows_as_tuple_string(content, spark)
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
    })



spark = SparkSession.builder \
    .appName("PostgreSQL with Spark") \
    .config("spark.jars", "./dependencies/postgresql-42.7.3.jar") \
    .getOrCreate()

create_views_for_query(URI, spark)

###################
### STREAMLIT #####
###################

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(
        content="""
        Hello! I'm a SQL assistant.\n
        Provide me with any questions you want for any information on the database 
        (including those indicated above)"""),
    ]


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


with st.expander("Write this sentences if you want more information...", expanded=True):
    st.write(
        """
        - What table and his attributes there are in this database?
        - Count the average number of rows for each tables on database.
        - Count the number of rows on database.
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
        spark.stop()
        st.stop()

    elif user_query.startswith("What table and his attributes"):
        with st.chat_message("Human"):
            st.markdown(user_query)
            llm = get_database_info(db=db)
            response = llm.invoke({})
            
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
                response = get_response(user_query, db, content, spark)
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