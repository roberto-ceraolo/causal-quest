from utils import add_sampled_suffix, sample_nq, download_file, get_first_user_content, is_english
import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login


def get_wc(): 
    """
    Get the WildChat dataset from the huggingface dataset hub. Needs to be run after huggingface-cli login because it uses the dataset library API, and the data is provided upon acceptance of the dataset license.
    
    """
    login()
    dataset = load_dataset("allenai/WildChat")
    sampled_dataset = dataset["train"].shuffle(seed=42).select(range(20000))
    sampled_dataset = sampled_dataset.map(lambda x: {'first_user_content': get_first_user_content(x['conversation'])})
    sampled_dataset = sampled_dataset.filter(lambda x: is_english(x['first_user_content']))
    sampled_dataset = sampled_dataset.shuffle(seed=42).select(range(5000))
    final_dataset = sampled_dataset.remove_columns([col for col in sampled_dataset.column_names if col not in ['conversation_id', 'first_user_content']])
    final_dataset = final_dataset.to_pandas()
    final_dataset = final_dataset.rename(columns={'conversation_id': 'wc_id', 'first_user_content': 'query'})
    return final_dataset

def get_nq(path):
    """
        Assumes the Natural Questions file is at the specified path.
    """

    output_path = add_sampled_suffix(path)
    sample_nq(path, output_path)
    df_nq = pd.read_json(output_path, lines=True)

    # if id in columns, rename it to query_id
    if 'example_id' in df_nq.columns:
        df_nq = df_nq.rename(columns={'example_id': 'nq_id'})
        
    df_nq = df_nq.rename(columns={'question_text': 'query'})

    return df_nq

def get_msmarco():
    # source: eval set v2.1 from https://microsoft.github.io/msmarco/#qna
    dataset = load_dataset("ms_marco", 'v2.1')
    df_msmarco = dataset["test"].to_pandas()
    df_msmarco = df_msmarco[["query", "query_id"]]
    df_msmarco.rename(columns={"query_id":"msmarco_id"}, inplace=True)
    return df_msmarco

    

def get_sg(output_folder_path):
    """
    Get the ShareGPT dataset from the huggingface dataset hub. The dataset library API is not working, so we download the files from the links. 
    """


    if not os.path.exists(output_folder_path + '/sg_90k_part1.json') or not os.path.exists(output_folder_path + 'sg_90k_part2.json'):
        print("ShareGPT files not found, downloading")
        urls = [
            'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1.json',
            'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2.json'
        ]
        for url in urls:
            download_file(url, output_folder_path + url.split('/')[-1])
    
    else:
        print("ShareGPT files found")
    
    df_sg_1 = pd.read_json(output_folder_path + '/sg_90k_part1.json')
    df_sg_2 = pd.read_json(output_folder_path + '/sg_90k_part2.json')
    df_sg = pd.concat([df_sg_1, df_sg_2], ignore_index=True)
    df_sg['query'] = df_sg['conversations'].apply(lambda x: next((item['value'] for item in x if item['from'] == 'human'), None))
    df_sg['query'] = df_sg['query'].astype(str)
    df_sg['query'] = df_sg['query'].convert_dtypes()
    df_sg = df_sg[df_sg['query'].str.split().apply(len) > 1]
    df_sg = df_sg.drop(columns=['conversations'])

    # if id in columns, rename it to query_id
    if 'id' in df_sg.columns:
        df_sg = df_sg.rename(columns={'id': 'sg_id'})

    return df_sg

def get_quora():
    quora_dataset = load_dataset("quora")
    # Get the train dataset
    train_dataset = quora_dataset["train"]

    # add an id col, using huggingface dataset
    train_dataset = train_dataset.add_column("id", range(len(train_dataset)))

    train_dataset = train_dataset.shuffle(seed=42).select(range(5000))

    # Extract the first element of the text field
    train_dataset = train_dataset.map(lambda example: {'text': example['questions']["text"][0]})

    # drop questions, is_duplicate
    train_dataset = train_dataset.remove_columns(["questions", "is_duplicate"])
    train_dataset = train_dataset.to_pandas()
    train_dataset = train_dataset.rename(columns={'text': 'query', 'id': 'quora_id'})

    return train_dataset

def get_labels(causalquest_path): 
    """
    Loads causalquest_labels.jsonl. 
    """

    # load causalquest_labels.jsonl from folder data
    df_labels = pd.read_json(causalquest_path, lines=True)
    df_labels['msmarco_id'] = df_labels['msmarco_id'].astype('Int64')
    return df_labels
