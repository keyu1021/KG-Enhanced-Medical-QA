import torch
import subprocess
import sys
from huggingface_hub import login
from datasets import load_dataset, Dataset
import os
import json
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format

from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

import warnings
warnings.filterwarnings("ignore")

token = os.getenv("HUGGINGFACE_TOKEN")

if token is None:
    raise ValueError("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN environment variable.")

login(token=token, add_to_git_credential=True)

def check_cuda_capability():
    """Check if the CUDA device meets the required compute capability."""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError('Hardware not supported for Flash Attention')
    else:
        raise RuntimeError('CUDA not available')


def install_packages_with_env_variable():
    """Install Python packages with an environment variable set."""
    # Set the MAX_JOBS environment variable for this process and its children.
    os.environ['MAX_JOBS'] = '4'

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja", "packaging"])
        # Now, MAX_JOBS is set for any subprocess started from here.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"])
    except subprocess.CalledProcessError as e:
        print("Failed to install packages:", e)
        raise

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
    check_cuda_capability()
    # install_packages()

    with open('./data/medical_graph_schema.json', 'r') as file:
        loaded_graph_schema = json.load(file)

    qa_df = pd.read_csv('./data/first_questions_and_cypher_queries.csv')
    dataset = create_cypher_qa_dataset(loaded_graph_schema, qa_df)

    system_message = """You are an text to Cypher query translator. Users will ask you questions in English and you will generate a Cypher query based on the provided Knowledge Graph SCHEMA.
    SCHEMA:
    {schema}"""

    dataset = dataset.shuffle().select(range(83))

    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    # dataset = dataset.train_test_split(test_size=0)

    print(dataset[1]["messages"])

    # save datasets to disk
    dataset.to_json("./data/train_dataset.json", orient="records")
    # dataset["test"].to_json("./data/test_dataset.json", orient="records")
    #
    # dataset = load_dataset("json", data_files="train_dataset.json", split="train")
    #
    model_id = "codellama/CodeLlama-7b-hf"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'  # to prevent warnings

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir="code-llama-7b-text-to-cypher",  # directory to save and repository id
        num_train_epochs=10,  # number of training epochs
        per_device_train_batch_size=3,  # batch size per device during training
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=1,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=True,  # push model to hub
        report_to="tensorboard",  # report metrics to tensorboard
    )

    max_seq_length = 3072  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save model
    trainer.save_model()
