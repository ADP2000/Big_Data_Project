# Big_Data_Project

## Overview
This is a Streamlit application that enhances SQL query generation using GPT-3 language models. The application connects to a PostgreSQL database and allows users to interact with a natural language interface to query database information.

## Setup Instructions
Follow these steps to set up and run the application:

### Prerequisites
1. Python 3.x installed on your system.
2. pip package manager installed.
3. Hadoop and Spark installed on your system (here is available a guide for the installation over Windows: "https://medium.com/@deepaksrawat1906/a-step-by-step-guide-to-installing-pyspark-on-windows-3589f0139a30")
4. PostgreSQL database instance with necessary access credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ADP2000/Big_Data_Project
cd Big_Data_Project
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
Set up environment variables:

1. Create a .env file in the root directory.

2. Add the following variable to .env:

```plaintext
GROQ_API_KEY = API_KEY_GROQ
```
replace API_KEY_GROQ with your api key groq available available via the groq cloud service

### Running the Application
1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Your default web browser will open with the application running. If not, visit http://localhost:8501 in your browser.

### Usage
- Upon running the application, you will see a form to enter your PostgreSQL database connection details (DB NAME, DB USER, DB PASSWORD, DB HOST, DB PORT).
- After submitting valid database connection details, you can interact with the natural language interface to query the database.
- Example queries you can try:
    - "What table and his attributes there are in this database?"
    - "Count the average number of rows for each tables on database."
    - "Count the number of rows on database."

### Additional Notes
- Ensure your PostgreSQL database is accessible from the network where you run this application.
- This application uses Streamlit for the web interface, SQLAlchemy for database connectivity, and Spark for SQL querying.



