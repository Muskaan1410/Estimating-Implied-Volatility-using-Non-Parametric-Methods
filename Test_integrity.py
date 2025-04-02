#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install psycopg2-binary pyyaml')


# In[2]:


config_content = """DATABASE:
  DB: "postgres"
  USERNAME: "postgres"
  PASSWORD: "your_password"
  HOST: "localhost"  
  PORT: "5432"  
"""

with open("config.yaml", "w") as file:
    file.write(config_content)

print("config.yaml created successfully!")


# In[3]:


import unittest
import yaml
import psycopg2

class TestOptionsDataSchema(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Establishes a connection to the database before running tests."""
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        cls.conn = psycopg2.connect(
            dbname=config['DATABASE']['DB'],
            user=config['DATABASE']['USERNAME'],
            password=config['DATABASE']['PASSWORD'],
            host=config['DATABASE']['HOST'],
            port=config['DATABASE']['PORT']
        )
        cls.cursor = cls.conn.cursor()

    @classmethod
    def tearDownClass(cls):
        """Closes the database connection after tests."""
        cls.cursor.close()
        cls.conn.close()
    
    def test_table_schemas(self):
    
        tables = ['options_data_SPY', 'options_data_SPX', 'options_data_TSLA']

        expected_schema = {
        'quote_unixtime': 'bigint',
        'quote_readtime': 'timestamp without time zone',
        'quote_date': 'date',
        'quote_time_hours': 'numeric',
        'underlying_last': 'numeric',
        'expire_date': 'date',
        'expire_unix': 'bigint',
        'dte': 'numeric',
        'c_delta': 'numeric',
        'c_gamma': 'numeric',
        'c_vega': 'numeric',
        'c_theta': 'numeric',
        'c_rho': 'numeric',
        'c_iv': 'numeric',
        'c_volume': 'numeric',
        'c_last': 'numeric',
        'c_size': 'text',
        'c_bid': 'numeric',
        'c_ask': 'numeric',
        'strike': 'numeric',
        'p_bid': 'numeric',
        'p_ask': 'numeric',
        'p_size': 'text',
        'p_last': 'numeric',
        'p_delta': 'numeric',
        'p_gamma': 'numeric',
        'p_vega': 'numeric',
        'p_theta': 'numeric',
        'p_rho': 'numeric',
        'p_iv': 'numeric',
        'p_volume': 'numeric',
        'strike_distance': 'numeric',
        'strike_distance_pct': 'numeric',
        'time_to_maturity_days': 'double precision',
        }

        for table in tables:
            print(f"Testing table: {table}") 
        
            with self.subTest(table=table):
                self.cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table.lower()}'
            """)
                schema = dict(self.cursor.fetchall())

                for column, expected_type in expected_schema.items():
                    self.assertEqual(schema.get(column), expected_type, f"{table}.{column} type mismatch.")

suite = unittest.TestLoader().loadTestsFromTestCase(TestOptionsDataSchema)
unittest.TextTestRunner().run(suite)


# In[ ]:




