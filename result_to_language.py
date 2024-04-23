from neo4j import GraphDatabase
import requests
import json
import openai

class Neo4jToChatGPTPipeline:
    def __init__(self, uri, user, password, chatgpt_api_key):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.api_key = chatgpt_api_key
    
    def close(self):
        self.driver.close()
    
    def get_query_result(self, cypher_query):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record for record in result]
    
    def send_to_chatgpt(self, question, query_result):
        # Create a stable prompt with the query result
        prompt = f"Please answer the medical question {question}, based on the result we get from our database, {query_result}."
        openai.api_key = self.api_key
        # Send the prompt to the ChatGPT API
           # Call the ChatGPT model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  #
            messages=[
                {"role": "system", "content": "You are an medical expert"},
                {"role": "user", "content": prompt}
            ]
        )
        
# Usage
uri = "bolt://34.205.141.191:7687"
user = "neo4j"
password = "miners-bit-gum"

pipeline = Neo4jToChatGPTPipeline(uri, user, password, chatgpt_api_key)

try:
    # example query
    question = "What drugs cause seizures?"
    cypher_query = "MATCH (e:Effect {name: 'Seizure'})-[:side_effect]->(d:Drug) RETURN d.name;"
    query_result = pipeline.get_query_result(cypher_query)
    print("Query result:", question, query_result)
    
    # Send the query result to ChatGPT
    chatgpt_response = pipeline.send_to_chatgpt(query_result)
    print("ChatGPT response:", chatgpt_response)
finally:
    print("database query failure...")
    pipeline.close()



