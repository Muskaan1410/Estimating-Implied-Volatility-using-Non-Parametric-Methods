#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install psycopg2-binary


# In[1]:


import psycopg2
import glob
import os

# Database connection details
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"

FILE_PATH = r"C:/Users/Muskaan Jain/OneDrive/Desktop/Finance Project/SPX_2023_TEST/*.txt"

try:
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="12345678",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # Get all .txt files in the directory
    file_list = glob.glob(FILE_PATH)

    if not file_list:
        raise Exception(" No files found in the directory!")

    # Iterate over each file and load into PostgreSQL
    for file in file_list:
        print(f" Loading file: {os.path.basename(file)}")

        with open(file, "r") as f:
            cur.copy_expert(f"COPY options_data_spy_test FROM STDIN DELIMITER ',' CSV HEADER NULL ' '", f)

        conn.commit()
        print(f" Successfully loaded: {os.path.basename(file)}")

    # Close the connection
    cur.close()
    conn.close()

    print(" All files successfully loaded into options_data_spy_test!")

except Exception as e:
    print(f" Error: {e}")


# In[ ]:




