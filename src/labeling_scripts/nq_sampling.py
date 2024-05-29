# Sampling procedure employed to gather datapoints from natural questions
# source of data: from https://ai.google.com/research/NaturalQuestions/download, simplified train data

import json
import random

def sample(path, output_path):
    # out of 307373 lines, we want sample of them randomly. 

    # we take 1 line every 60 lines on average
    sampling_rate = 60
    max_samples = 5000

    # set the seed 
    random.seed(42)

    with open(path, "r") as f:
        with open(output_path, "w") as f_out:
            sample_count = 0
            for i, line in enumerate(f):
                if random.random() < 1 / sampling_rate:
                    f_out.write(line)
                    sample_count += 1
                if i % 1000 == 0:
                    print(f"Processed {i} lines")
                if sample_count >= max_samples:
                    break
    print("Done")


if __name__ == "__main__":
    sample()