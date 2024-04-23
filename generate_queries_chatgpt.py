from neo4j import GraphDatabase
import requests
import json
import openai

class Neo4jToChatGPTPipeline:
    def __init__(self, uri, user, password, chatgpt_api_key, schema):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.api_key = chatgpt_api_key
        self.schema = schema
    
    def close(self):
        self.driver.close()
    
    # def get_query_result(self, cypher_query):
    #     with self.driver.session() as session:
    #         result = session.run(cypher_query)
    #         return [record for record in result]
    
    def send_to_chatgpt(self, questions):
        # Create a stable prompt with the query result
        prompt = f"Please generate the corresponding Cypher query based on this graph schema {self.schema} for this question {questions}. Make sure to use only entities and relationships defined in the schema"
        openai.api_key = self.api_key
        # Send the prompt to the ChatGPT API
           # Call the ChatGPT model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  #
            messages=[
                {"role": "system", "content": "You are ann expert in Cypher graph queries."},
                {"role": "user", "content": prompt}
            ]
        )
        
# Usage
uri = "bolt://34.205.141.191:7687"
user = "neo4j"
password = "miners-bit-gum"
chatgpt_api_key = "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"

schema = '''The medical knowledge graph that consists of 5 types of nodes 
        Drug, disease, gene_protein, anatomy and effect With following relationships:
        Drug to disease: contraindication
        Drug to disease: indication
        drug to disease: off_label_use
        drug to drug: synergy_interaction

        effect to drug: side_effect
        gene_protein to drug: transporter
        gene_protein to drug: carrier
        gene_protein to drug: enzyme
        gene_protein to drug: target

        disease to disease: parent_child

        gene_protein to disease: associated_with
        effect to gene_protein: associated with
        gene_protein to anatomy: expression_absent
        gene_protein to anatomy: expression_present'''

questions = [
    "What drugs are recommended for gastroesophageal reflux disease based on gene_protein targets?",
    "What are the side effects associated with the prescribed drugs?",
    "What are the drugs indicated for obesity disorder that have interactions with gene_proteins acting as enzymes?"
]


pipeline = Neo4jToChatGPTPipeline(uri, user, password, chatgpt_api_key, schema)

#Send the result to ChatGPT
chatgpt_response = pipeline.send_to_chatgpt(questions=questions)
print("ChatGPT response:", chatgpt_response)

# try:
#     # Cypher query to fetch data from the Neo4j graph database
#     cypher_query = "MATCH (e:Effect {name: 'Seizure'})-[:side_effect]->(d:Drug) RETURN d.name;"
#     query_result = pipeline.get_query_result(cypher_query)
#     print("Query result:", query_result)
    
#     # Send the query result to ChatGPT
#     chatgpt_response = pipeline.send_to_chatgpt(query_result)
#     print("ChatGPT response:", chatgpt_response)
# finally:
#     print("database query failure...")
#     pipeline.close()



