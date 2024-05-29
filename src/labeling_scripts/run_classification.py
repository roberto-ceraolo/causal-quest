# Script to run classification of CausalQuest using the normal OpenAI API

import jsonlines
import sys
from utils import get_cat1_outcome, get_cat2_outcome, get_prompt_cat_1_iteration_3_CoT, get_prompt_cat_1_iteration_3, get_prompt_cat_1_iteration_4, get_prompt_cat_1_iteration_4_cot, get_prompt_cat_1_iteration_5, get_prompt_cat_1_iteration_6
import pandas as pd
import os
import argparse
import time

feature_names = {
    "subjectivity": "is_subjective",
    "domain": "domain_class", 
    "action": "action_class",
    "causal": "is_causal"
}

cat2_names = ["subjectivity", "domain", "action"]

def run(classification_type, input_path, output_path, model, prompt_function=None, system_prompt_flag=True):
    """
    Run the classification for the given classification type

    Args:
    classification_type: str, the type of classification to run. Possible values: "subjectivity", "action", "domain", "causal"

    Returns:
    None

    """
    
    if input_path.endswith('.jsonl'):
        db = pd.read_json(input_path, lines=True)
    elif input_path.endswith('.xlsx'):
        db = pd.read_excel(input_path)
    else:
        print("Unsupported file format. Please provide a JSONL or XLSX file.")
        return
    

    if os.path.exists(output_path):
        with jsonlines.open(output_path, mode='r') as reader:
            # get the number of lines
            lines = list(reader)
            if len(lines) == 0:
                print("The file is empty. Proceeding")
            else:
                # take the query of the last processed line
                last_query = lines[-1]['query']
                print(f"Number of existing lines: {len(lines)}")

                # check that the last line of db is the previous instance of the last line
                if db.iloc[len(lines)-1]['query'] != last_query:
                    print("The last line of the json file is not the same as the last line of the dataframe. Exiting")
                    return
                else:
                    print("The last line of the json file is the same as the last line of the dataframe. Proceeding")

                db = db.iloc[len(lines):].copy()


    if classification_type not in cat2_names:
        cat1 = True
    else:
        cat1 = False

    print(f"To be processes: {len(db)} rows")

    # Open the JSON file in append mode
    with jsonlines.open(output_path, mode='a') as writer:
        # Generate JSON lines on the fly and append them to the file
        for i, row in db.iterrows():
            data = {
                "source": row["source"],
                "query": row["query"],
                "summary": row["summary"],
                "id": row["id"],
                "is_causal": row["is_causal"] if "is_causal" in row else None,
                "is_subjective": row["is_subjective"] if "is_subjective" in row else None,
                "domain_class": row["domain_class"] if "domain_class" in row else None,
                "action_class": row["action_class"] if "action_class" in row else None,
                "is_causal_raw": row["is_causal_raw"] if "is_causal_raw" in row else None,
                "is_subjective_raw": row["is_subjective_raw"] if "is_subjective_raw" in row else None,
                "domain_class_raw": row["domain_class_raw"] if "domain_class_raw" in row else None,
                "action_class_raw": row["action_class_raw"] if "action_class_raw" in row else None
            }

            if cat1: 
                outcome, raw_outcome = get_cat1_outcome(row, model, prompt_function_name)
            else:
                if row["is_causal"] == False:
                    continue
                outcome, raw_outcome = get_cat2_outcome(row, model, prompt_function_name, classification_type, system_prompt_flag)

            feature_key = feature_names[classification_type]
            data[feature_key] = outcome
            data[f"{feature_key}_raw"] = raw_outcome

            # Write the data as a JSON line to the file
            writer.write(data)
            print(f"Processed {i+1} rows")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification on a dataset')
    parser.add_argument('input_path', type=str, help='Path to the input dataset')
    parser.add_argument('classification_type', type=str, help='Type of classification to run. Possible values: "subjectivity", "action", "domain", "causal"')
    parser.add_argument('model', type=str, help='OpenAI model to use. Model used: "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo"')
    parser.add_argument('prompt_function', type=str, help='Name of the prompt function to use among the ones in utils.py')
    args = parser.parse_args()
    
    input_path, model, classification_type, prompt_function_name = args.input_path, args.model, args.classification_type, args.prompt_function
    output_folder_path = os.path.join(os.path.dirname(input_path), "prompt_engineering")
    file_name  = os.path.splitext(os.path.basename(input_path))[0]
    system_prompt_flag=False


    print(f"Running classification type {classification_type}, normal API, prompt function {prompt_function_name}, model {model}\n")
    output_path = f"{output_folder_path}/{file_name}_{classification_type}_{prompt_function_name}_{model}_sysprompt_{system_prompt_flag}.jsonl"
    print(f"Output file {output_path}\n")

    # if output path exists, stop
    if os.path.exists(output_path):
        print("Output file already exists. Exiting")
        sys.exit(1)
    
    run(classification_type, input_path, output_path, model, system_prompt_flag=system_prompt_flag)
    
