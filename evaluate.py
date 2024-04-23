import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset, Dataset
from random import randint
from tqdm import tqdm
import json
import os
import pandas as pd
import csv


def get_folder_size(folder_path):
    """
    Calculate the total size of a folder including all its subfolders and files.

    Args:
    - folder_path: A string representing the path to the folder.

    Returns:
    - The total size of the folder in bytes.
    """
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path):  # Skip if it's a symbolic link
                total_size += os.path.getsize(file_path)
    return total_size

def evaluate(sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    if predicted_answer == sample["messages"][2]["content"]:
        return 1
    else:
        return 0

def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message.format(schema=sample["context"])},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }


def create_cypher_qa_dataset(graph_schema, qa_df):

    dataset = {
        "features": ['question', 'answer', 'context'],
        "num_rows": len(qa_df),
        "data": []
    }

    # Populate the dataset with questions, answers, and the schema as context
    for index, row in qa_df.iterrows():
        dataset['data'].append({
            "question": row['Question'],
            "answer": row['Cypher Query'],
            "context": graph_schema
        })

    df = pd.DataFrame(dataset['data'])
    # Convert the pandas DataFrame to a Hugging Face Dataset object
    hf_dataset = Dataset.from_pandas(df)

    return hf_dataset


if __name__ == "__main__":

    folder_path = './code-llama-7b-text-to-cypher'
    print(f"Total size of '{folder_path}' is: {get_folder_size(folder_path)  / (1024**3)} GB")

    peft_model_id = "./code-llama-7b-text-to-cypher"
    # peft_model_id = args.output_dir

    # Load Model with PEFT adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    # load into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load our test dataset
    # eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

    with open('./data/medical_graph_schema.json', 'r') as file:
        loaded_graph_schema = json.load(file)

    qa_df = pd.read_csv('./data/test_questions_and_cypher_queries.csv')
    dataset = create_cypher_qa_dataset(loaded_graph_schema, qa_df)

    system_message = """You are an text to Cypher query translator. Users will ask you questions in English and you will generate a Cypher query based on the provided Knowledge Graph SCHEMA.
    SCHEMA:
    {schema}"""

    dataset = dataset.shuffle().select(range(16))
    eval_dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

    print(eval_dataset[1]["messages"])

    eval_dataset.to_json("./data/test_dataset.json", orient="records")

    rand_idx = randint(0, len(eval_dataset))

    # Test on sample
    prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False,
                                                add_generation_prompt=True)

    print(prompt)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1,
                   eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
    print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

    success_rate = []
    number_of_eval_samples = 16
    # iterate over eval dataset and predict
    for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
        success_rate.append(evaluate(s))

    # compute accuracy
    accuracy = sum(success_rate) / len(success_rate)

    print(f"Accuracy: {accuracy * 100:.2f}%")