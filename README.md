# KG-Enhanced-Medical-QA
COMP545 Final Project


## Description
This project integrates a knowledge graph into a question-answering system leveraging a fine-tuned language model. It utilizes a comprehensive dataset of medical knowledge encompassing diseases, drugs and their interrelationships etc. The original data for this project is obtained from Harvar Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM

## Graph Database
We have seperated the datasets for different nodes and edges available separately for easy upload. Please use the following drive to download the datasets:
https://drive.google.com/drive/folders/1tB41dGLulG7q5CD1qq4UgFPZ6XBUv_VT?usp=drive_link
<img width="587" alt="Screenshot 2024-04-23 at 19 48 07" src="https://github.com/keyu1021/KG-Enhanced-Medical-QA/assets/100107070/c1f69091-7133-4d37-90b6-4f4824a80ed1">


## Scripts
load_neo4j.py: load the csv files into the graph database
generate_queries_chatgpt.py: baseline model GPT4, generate 
finetune.py: fine tune code llama
evaluate.py: evaluate our proposed model along with baseline models
generate_queries_chatgpt: baseline model GPT4 on generate queries based on questions
result_to_language.py: convert the result from query into an answer using natural language

## Datasets
