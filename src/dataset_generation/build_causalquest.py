# script to generate the labeled causalquest

import os
PATH = ""
os.environ['HF_DATASETS_CACHE'] = PATH

import argparse
from dataloaders import get_msmarco, get_sg, get_quora, get_nq, get_labels, get_wc


def get_causalquest(nq_path, causalquest_path, output_path):
    """
    Generate the labeled causalquest dataset by merging the labels with queries from the different sources.

    Args:
        nq_path (str): Path to the downloaded Natural Questions file. In the same folder we download sg. The other components all go to hf cache. 

    Returns:    
        None

    """

    folder_path = os.path.dirname(nq_path)
    
    
    # get wc data
    print("Getting WildChat data...")
    df_wc = get_wc()
    print("Done")

    # get nq data
    print("Getting Natural Questions data...")
    df_nq = get_nq(nq_path)
    print("Done")

    # get msmarco data
    print("Getting MS Marco data...")
    df_msmarco = get_msmarco()
    print("Done")

    # get sg data
    print("Getting ShareGPT data...")
    df_sg = get_sg(folder_path)
    print("Done")

    
    # get quora data
    print("Getting Quora data...")
    df_quora = get_quora()
    print("Done")
    
    print("Merging data with CausalQuest labels...")
    # get labels data
    df_labels = get_labels(causalquest_path)

    df_labels["query"] = ""
    merge_info = [
            ('sg_id', df_sg),
            ('wc_id', df_wc),
            ('nq_id', df_nq),
            ('msmarco_id', df_msmarco),
            ('quora_id', df_quora)
        ]

    # Augment df_labels with queries from all other sources by matching on the specified columns
    for label_col, df in merge_info:
        if "query_id" in df.columns:
            df = df.rename(columns={'query_id': f'{label_col}'})

        df_labels = df_labels.merge(df[[label_col, 'query']], on=label_col, how='left', suffixes=('', f'_{label_col.replace("_id", "")}'))


    columns = ['query_sg', 'query_wc', 'query_nq', 'query_msmarco', 'query_quora']
    df_labels['query'] = df_labels[columns].bfill(axis=1).iloc[:, 0]    
    df_labels.drop(columns=columns, inplace=True)

    # save the final dataframe to a jsonl file
    df_labels.to_json(output_path + '/causalquest.jsonl', orient='records', lines=True)
    print("Done")
    print(f"Saved the final dataframe to {output_path + '/causalquest.jsonl'}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labeled causalquest")
    parser.add_argument("nq_path", type=str, help="Path to the downloaded Natural Questions file")
    parser.add_argument("causalquest_path", type=str, help="Path to the CausalQuest labels file")
    parser.add_argument("--output_path", type=str, help="Path to save the labeled causalquest file")
    args = parser.parse_args()

    get_causalquest(args.nq_path, args.causalquest_path, args.output_path)