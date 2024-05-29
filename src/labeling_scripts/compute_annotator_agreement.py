# Compute annotator agreement or classification report given 2 xls files with annotations
# Usage: python3 compute_agreement.py -a1 <file1> -a2 <file2> -f <feature> -mode <mode>

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import classification_report

feature_names_map = {"action": "action_class", "domain":"domain_class", "subjectivity":"is_subjective", "causal":"is_causal"}

import argparse
import sys


def read_xls(file):
    df = pd.read_excel(file, index_col=0)
    if 'is causal?' in df.columns:
        df.rename(columns={'is causal?': 'is_causal'}, inplace=True)
    return df

def read_jsonl(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return data

def read_csv(file): 
    # read csv file with id and feature
    df = pd.read_csv(file, index_col=0)
    df["id"] = df.index
    return df



def compute_agreement(file1, file2, mode = "agreement", feature="is_causal", verbose=False):
    if file1.endswith(".xls"):
        df1 = pd.read_excel(file1)[['id', "summary", feature]]
    elif file1.endswith(".jsonl"):
        df1 = pd.read_json(file1, lines=True)[['id', "summary", feature]]
    elif file1.endswith(".csv"):
        df1 = pd.read_csv(file1)[['id', "summary", feature]]
    else:
        print("Error: file format not supported")
        sys.exit(1)

    if file2.endswith(".xls"):
        df2 = pd.read_excel(file2)
    elif file2.endswith(".jsonl"):
        df2 = pd.read_json(file2, lines=True)
    elif file2.endswith(".csv"):
        df2 = pd.read_csv(file2)
    else:
        print("Error: file format not supported")
        sys.exit(1)

    df2 = df2[['id', "summary", feature, f"{feature}_annotator_1", f"{feature}_annotator_2"]]


    df1 = df1.dropna()
    df2 = df2.dropna()
    if len(df1) != len(df2):
        print("Error: the two files have different number of annotations")
        sys.exit(1)

    # sort by id
    df1 = df1.sort_values(by='id').reset_index(drop=True)
    df2 = df2.sort_values(by='id').reset_index(drop=True)
    
    # Check that id matches for each line
    if not df1['id'].equals(df2['id']):
        print("Error: the two files have different ids")
        sys.exit(1)
    y_gpt = df1[feature].values
    y_ground_truth = df2[feature].values

    y_annotator1 = df2[f"{feature}_annotator_1"].values
    y_annotator2 = df2[f"{feature}_annotator_2"].values
    

    if isinstance(y_gpt[0], np.bool_) or isinstance(y_ground_truth[0], np.bool_):
        y_gpt = y_gpt.astype(str)
        y_ground_truth = y_ground_truth.astype(str)
        y_annotator1 = y_annotator1.astype(str)
        y_annotator2 = y_annotator2.astype(str)
    
    if feature == "is_subjective": 
        y_ground_truth = ["True" if val == "Subjective" else "False" for val in y_ground_truth]
        y_annotator1 = ["True" if val == "Subjective" else "False" for val in y_annotator1]
        y_annotator2 = ["True" if val == "Subjective" else "False" for val in y_annotator2]

    # compute fleiss agreement
    cohen_kappa = None
    if mode == "agreement":
        cohen_kappa = metrics.cohen_kappa_score(y_gpt, y_ground_truth)
        report_ground_truth = None
    else:
        report_ground_truth = classification_report(y_gpt, y_ground_truth, output_dict=True)
        cohen_kappa = None
    
    # compute the number of disagreements
    disagreements = 0
    disagreement_ids = []
    # use iterrows to get the index of the row
    for index, row in df1.iterrows():
        if str(row[feature]) != str(df2.loc[index, feature]):
            disagreements += 1
            disagreement_ids.append(row['id'])
            if verbose: 
                print("Disagreement: ", row['summary'])
                print("Annotator 1: ", row[feature])
                print("Annotator 2: ", df2.loc[index, feature])
                print("\n")

    if verbose: 
        print("Disagreements: ", disagreements)
        print("Disagreement ids: ", disagreement_ids)

    return report_ground_truth, cohen_kappa




def main():
    parser = argparse.ArgumentParser(description='Compute annotator agreement')
    parser.add_argument('-a1', '--annotator1', help='file with annotations from annotator 1', required=True)
    parser.add_argument('-a2', '--annotator2', help='file with annotations from annotator 2', required=True)
    parser.add_argument('-f', '--feature', help='feature to compute agreement for', required=False)
    parser.add_argument('-mode', '--mode', help='either annotator or report', required=False)
    args = parser.parse_args()
    
    print("File 1: ", args.annotator1)
    print("File 2: ", args.annotator2)

    chosen_feature = feature_names_map[args.feature]
    
    report, agreement = compute_agreement(args.annotator1, args.annotator2, chosen_feature, verbose=True)
    print(f"Feature: {chosen_feature}")
    if agreement:
        print("Agreement (Cohen's kappa): ", agreement)
    if report:
        print("Classification report with groung truth: ", report)


if __name__ == "__main__":
    main()
    