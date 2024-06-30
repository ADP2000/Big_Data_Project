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

# Function to load environment variables
load_dotenv()

# Function to execute query and get rows as tuple string
def execute_query_and_get_rows_as_tuple_string(query, spark):
    df = spark.sql(query)
    result = df.collect()
    columns = df.columns
    rows_as_tuples = [tuple(row[col] for col in columns) for row in result]
    result_str = str(rows_as_tuples)
    return result_str

# Function to create views for query
def create_views_for_query(spark, db_host, db_port, db_name, db_user, db_password):
    jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}"
    properties = {
        "user": db_user,
        "password": db_password,
        "driver": "org.postgresql.Driver"
    }

    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
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

# Function to get database info
def get_database_info(db):
    template = """
    You are an assistant tasked with providing a user with information about a specific database. The template you need to use is similar to this:
        
        Hello! I'm an information assistant.

        The dataset represents [Brief Description of the Database]. Here is a detailed overview of the tables and their attributes in the database:

        <SCHEMA>{schema}</SCHEMA>
    
    You must always give me the name of the database tables and what the table attributes indicate based on the schema provided.
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

# Function to get SQL chain
def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query for a PostgreSQL database, that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
        
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    You have to use ONLY this table and not others. The information shcema table is not available.
    
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

# Function to get response
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
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
        schema=lambda _: db.get_table_info(),
        response=lambda _: execute_query_and_get_rows_as_tuple_string(content, spark)
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"question": user_query})

# Initialize SparkSession
def initialize_spark():
    return SparkSession.builder \
        .appName("PostgreSQL with Spark") \
        .config("spark.jars", "./dependencies/postgresql-42.7.3.jar") \
        .getOrCreate()

# Main app
def main():
    st.set_page_config(page_title="Query Augmented Generation", page_icon="llm.jpg", layout="wide")

    if "db_info" not in st.session_state:
        st.session_state.db_info = None

    error_message = ""

    if st.session_state.db_info is None:
        with st.form("db_form"):
            st.title("Database Connection")
            db_name = st.text_input("DB NAME")
            db_user = st.text_input("DB USER")
            db_password = st.text_input("DB PASSWORD", type="password")
            db_host = st.text_input("DB HOST")
            db_port = st.text_input("DB PORT")
            submitted = st.form_submit_button("CONNECT", use_container_width=True)

            if submitted:
                try:
                    URI = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"
                    db = SQLDatabase.from_uri(URI)
                    # Test connection
                    _ = db.get_table_info()
                    st.session_state.db_info = {
                        "DB_NAME": db_name,
                        "DB_USER": db_user,
                        "DB_PASSWORD": db_password,
                        "DB_HOST": db_host,
                        "DB_PORT": db_port
                    }
                    st.experimental_rerun()
                except Exception as e:
                    error_message = str(e)

        if error_message:
            st.error(f"Connection failed: {error_message}")

        # Add CSS for styling
        st.markdown(
            """
            <style>
                .stTextInput label {
                    font-weight: bold;
                    text-transform: uppercase;
                }
                .stTextInput input {
                    font-size: 1rem;
                }
                .stButton button {
                    font-size: 1.2rem;
                    display: block;
                    margin: 0 auto;
                    transition: background-color 0.3s ease;
                }
                .stButton button:hover {
                    background-color: red;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    else:
        db_info = st.session_state.db_info
        DB_NAME = db_info["DB_NAME"]
        DB_USER = db_info["DB_USER"]
        DB_PASSWORD = db_info["DB_PASSWORD"]
        DB_HOST = db_info["DB_HOST"]
        DB_PORT = db_info["DB_PORT"]

        URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        db = SQLDatabase.from_uri(URI)
        spark = initialize_spark()
        create_views_for_query(spark, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(
                    content="""
                    Hello! I'm a SQL assistant.\n
                    Provide me with any questions you want for any information on the database 
                    (including those indicated above)"""),
            ]

        col1, col2 = st.columns([8, 1])
        with col1:
            st.title("Query Augmented Generation")
        
        with st.expander("Write this sentences if you want more information...", expanded=True):
            st.write(
                """
                - What table and his attributes there are in this database?
                - Count the average number of rows on database.
                - Count the total number of rows on database.
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

            st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()
