# scripts to download, sample and preprocess the components of CausalQuest

from datasets import load_dataset
import pandas as pd
import pycld2 as cld2
import random

def is_english(text):
    try:
        is_reliable, _, details = cld2.detect(text)
        return details[0][1] == 'en'
    except:
        return False



def get_shareGPT(in_folder, out_folder):
    # dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered") # -> does not work
    # we download sg_90k_part1.json and sg_90k_part2.json
    # wget "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1.json"
    # wget "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2.json"

    df_sg_1 = pd.read_json(f"{in_folder}/sg_90k_part1.json")
    df_sg_2 = pd.read_json(f"{in_folder}/sg_90k_part2.json")
    df_sg = pd.concat([df_sg_1, df_sg_2], ignore_index=True)
    df_sg['query'] = df_sg['conversations'].apply(lambda x: next((item['value'] for item in x if item['from'] == 'human'), None))
    df_sg['query'] = df_sg['query'].astype(str)
    df_sg['query'] = df_sg['query'].convert_dtypes()
    df_sg = df_sg[df_sg['query'].str.split().apply(len) > 1]
    df_sg = df_sg[df_sg['query'].apply(lambda x: is_english(x))]
    df_sg = df_sg.sample(5000, random_state=42)
    df_sg = df_sg.drop(columns=['conversations'])
    df_sg.rename(columns={"id": "query_id"}, inplace=True)
    df_sg["source"] = "sg"
    df_sg.to_json(f"{out_folder}/sg_sample_5000.jsonl", orient="records", lines=True)

def get_wildchat(out_folder):
    # If the dataset is gated/private, make sure you have run huggingface-cli login
    from huggingface_hub import login
    login()
    dataset = load_dataset("allenai/WildChat")
    sampled_dataset = dataset["train"].shuffle(seed=42).select(range(20000))

    sampled_dataset = sampled_dataset.map(lambda x: {'first_user_content': get_first_user_content(x['conversation'])})
    sampled_dataset = sampled_dataset.filter(lambda x: is_english(x['first_user_content']))
    sampled_dataset = sampled_dataset.shuffle(seed=42).select(range(5000))
    final_dataset = sampled_dataset.remove_columns([col for col in sampled_dataset.column_names if col not in ['conversation_id', 'first_user_content']])
    final_dataset = final_dataset.to_pandas()
    final_dataset["source"] = "wc"
    final_dataset = final_dataset.rename(columns={'conversation_id': 'query_id', 'first_user_content': 'query'})
    final_dataset.to_json(f"{out_folder}/wc_sample_5000.jsonl", orient="records", lines=True)

def get_first_user_content(conversation):
    for message in conversation:
        if message['role'] == 'user':
            return message['content']
    return None  # Return None or appropriate value if no user role is found

    

def get_msmarco(in_file, out_folder):
    # source: eval set v2.1 from https://microsoft.github.io/msmarco/#qna

    df_marco = pd.read_json(in_file)

    # Set the seed
    random.seed(42)

    # Sample randomly 5000 rows
    df_sample_marco = df_marco.sample(n=5000)
    df_sample_marco["source"] = "msmarco"
    df_sample_marco = df_sample_marco[["query", "source", "query_id"]]
    # reset indices
    df_sample_marco.reset_index(drop=True, inplace=True)

    # save it as a jsonl file
    df_sample_marco.to_json(f"{out_folder}/msmarco_sample_5000.jsonl", orient="records", lines=True)


def get_natural_questions(in_file, out_folder):
    # the file is generated via nq_sampling.py

    df_nq = pd.read_json(in_file, lines=True)

    df_nq["source"] = "nq"

    df_nq = df_nq[["source", "question_text", "example_id"]]

    df_nq.rename(columns={"question_text": "query", "example_id": "query_id"}, inplace = True)

    df_nq.to_json(f"{out_folder}/nq_sample_5000.jsonl", orient="records", lines=True)

def get_quora(out_folder):
    quora_dataset = load_dataset("quora")
    # Get the train dataset
    train_dataset = quora_dataset["train"]

    # Sample randomly 5000 rows
    train_dataset = train_dataset.shuffle(seed=42).select(range(5000))

    # Extract the first element of the text field
    train_dataset = train_dataset.map(lambda example: {'text': example['questions']["text"][0]})

    # drop questions, is_duplicate
    train_dataset = train_dataset.remove_columns(["questions", "is_duplicate"])


    train_dataset = train_dataset.to_pandas()
    train_dataset["source"] = "quora"

    # add a col with the index as id
    train_dataset["id"] = train_dataset.index
    train_dataset = train_dataset.rename(columns={'text': 'query', 'id': 'query_id'})

    train_dataset.to_json(f"{out_folder}/quora_sample_5000.jsonl", orient="records", lines=True)


def merge(path_components_folder):
    import glob

    # Get a list of all JSONL files in the folder
    jsonl_files = glob.glob(path_components_folder + "*.jsonl")

    # Initialize an empty list to store the DataFrames
    dfs = []

    # Iterate over each file
    for file in jsonl_files:
        # Read the JSONL file as a pandas DataFrame
        df = pd.read_json(file, lines=True)
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    # check for duplicates
    duplicates = combined_df.duplicated(subset=["query"])
    # drop them
    combined_df = combined_df.drop_duplicates(subset=["query"])
