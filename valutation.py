import json
import os

def append_to_json(user_input, sql_query, system_output, execution_time, file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = []

    new_entry = {
        "user_input": user_input,
        "sql_query": sql_query,
        "system_output": system_output,
        "execution_time": execution_time
    }

    data.append(new_entry)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)  # indent per leggibilità
