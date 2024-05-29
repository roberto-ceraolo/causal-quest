# script to run the categorisation of CausalQuest using OpenAPI BatchAPI. The full chunking procedure is needed only for rate-limited accounts. Otherwise the full dataset can be sent at once. 


# input: the file with the categories to be filled
# it gets transformed in the input required by openAI
# gets divided in chunk of 150k tokens
# the files are all uploaded to openai
# then a function sends one by one the requests to openai
# finally, the outputs are retrieved and processed

import datetime
import tiktoken
import json
import os
from utils import get_system_prompt_cat_2,process_output_CoT, get_prompt_cat_1_iteration_4
from openai import OpenAI
import time

source_mapping = {
    "sg": "ChatGPT",
    "wc": "ChatGPT",
    "quora": "Quora",
    "nq": "Google",
    "msmarco": "Bing"
}


def convert_to_openai_input(file_name, output_folder, cat, prompt_function, model):
    # transform the data in the format required by openai
    cat1 = False
    # if the file already exists, return it
    formatted_file = os.path.join(output_folder, f'openai_batch_job_{cat}.jsonl')
    # if os.path.exists(formatted_file):
    #     print(f"File {formatted_file} already exists, returning it")
    #     return formatted_file

    if cat == "causal":
        cat1 = True

    written_lines_counter = 0
    with open(file_name, 'r') as infile, open(formatted_file, 'w') as outfile:
        for i, line in enumerate(infile):
            data = json.loads(line)
            
            if not cat1:
                is_causal = data["is_causal"]
                if is_causal == False: 
                    continue

            summary = data["summary"]
            source = source_mapping[data['source']]
            if cat1: 
                system_prompt = None
            else:     
                system_prompt = get_system_prompt_cat_2()

            prompt = prompt_function(source, summary)

            if system_prompt: 
                output_data = {
                    "custom_id": f"request_{written_lines_counter}_queryId_{data['id']}_source_{data['source']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    }
                }
            else:
                output_data = {
                    "custom_id": f"request_{written_lines_counter}_queryId_{data['id']}_source_{data['source']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                }
            json.dump(output_data, outfile)
            written_lines_counter += 1
            print(f"Written {written_lines_counter} lines")
            outfile.write('\n')
    return formatted_file
    
    
def chunkify(formatted_file, output_folder, n_tokens, model):
    # divide the data in chunks of chunk_size tokens

    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model(model)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # create a folder chunks in the output folder
    output_base_path = os.path.join(output_folder, "chunks")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    else: 
        return output_base_path
    

    current_chunk = 0
    current_token_count = 0
    chunk_data = []

    with open(formatted_file, 'r') as infile:
        for line in infile:
            # Parse the JSON line
            data = json.loads(line)
            # Process each message in the body
            for message in data['body']['messages']:
                tokens = enc.encode(message['content'])
                token_count = len(tokens)
                current_token_count += token_count
            
            # Check if the current token count has exceeded the limit
            if current_token_count >= n_tokens:
                # Write the current list of messages to a file
                output_filename = os.path.join(output_base_path, f'chunk_{current_chunk}.jsonl')
                with open(output_filename, 'w') as outfile:
                    for item in chunk_data:
                        json.dump(item, outfile)
                        outfile.write('\n')
                # print the number of tokens
                print(f"Chunk {current_chunk} has {current_token_count} tokens.")

                # Reset for the next chunk
                chunk_data = []
                current_chunk += 1
                current_token_count = token_count  # Start new count with the current message

            # Add the data to the current chunk's data list
            chunk_data.append(data)

    # Handle any remaining data that didn't reach the token limit
    if chunk_data:
        output_filename = os.path.join(output_base_path, f'chunk_{current_chunk}.jsonl')
        with open(output_filename, 'w') as outfile:
            for item in chunk_data:
                json.dump(item, outfile)
                outfile.write('\n')
        print(f"Chunk {current_chunk} has {current_token_count} tokens.")
    return output_base_path



def delete_all_files_in_openai(client):
    # optionally, delete all the files in the openai before starting
    # deletes all files on openAI

    while client.files.list().data:
        file_id = client.files.list().data[0].id
        client.files.delete(file_id)
        print(f"Deleted file with ID: {file_id}")
    assert len(client.files.list().data) == 0


def files_upload_to_openai(chunk_path, client):
    # upload all the files to openai

    dicts_path = os.path.dirname(chunk_path)
    
    # if in chunk_path there are already id_to_filename and filename_to_id, load them
    if os.path.exists(os.path.join(dicts_path, "filename_to_id.json")):
        with open(os.path.join(dicts_path, "filename_to_id.json"), "r") as file:
            filename_to_id = json.load(file)
        with open(os.path.join(dicts_path, "id_to_filename.json"), "r") as file:
            id_to_filename = json.load(file)
        return filename_to_id, id_to_filename
    
    filename_to_id = {}
    id_to_filename = {}

    for filename in os.listdir(chunk_path):

        file = client.files.create(
        file=open(os.path.join(chunk_path, filename), "rb"),
        purpose="batch"
        )

        file_id = file.id
        filename_to_id[filename] = file_id
        id_to_filename[file_id] = filename
        print(f"Uploaded {filename} with ID: {file_id}")
        
    # save the two dicts in a file
    with open(os.path.join(dicts_path, "filename_to_id.json"), "w") as file:
        json.dump(filename_to_id, file)
    with open(os.path.join(dicts_path, "id_to_filename.json"), "w") as file:
        json.dump(id_to_filename, file)

    return filename_to_id, id_to_filename


def send_requests_to_openai(client,filename_to_id, folder_path):
    """
        Takes file per file in a local folder, retrieves the id that OpenAI assigned to the file when I uploaded it, and then 
        sends the request to OpenAI.
    
    """

    # create a log.txt file 
    log_file_path = f"{os.path.dirname(folder_path)}/log.txt"
    if not os.path.exists(log_file_path):
        log_file = open(log_file_path, "w")
        log_file.write("Log of batch jobs\n\n")
        log_file.write("Filename to ID mapping:\n")
        for filename, file_id in filename_to_id.items():
            log_file.write(f"{filename}: {file_id}\n")
        log_file.close()


    # Get the list of files in the folder
    files = os.listdir(folder_path)

    missing_chunk_numbers = []
    for file in files:
        if "jsonl" in file and "_sent" not in file:
            file = file[:-6]
            chunk_n = file.split("_")[1]
            missing_chunk_numbers.append(int(chunk_n))
    
    if len(missing_chunk_numbers) == 0:
        print("All chunks have been sent to OpenAI")
        return
    
    # sort 
    missing_chunk_numbers.sort()

    for chunk_n in missing_chunk_numbers:
        print(f"Sending chunk {chunk_n}")
        batch_id = run_chunk(chunk_n, client, filename_to_id, chunk_path, log_file_path)
        # check status
        status = client.batches.retrieve(batch_id).status
        while status == "in_progress" or status == "finalizing":
            time.sleep(30)
            status = client.batches.retrieve(batch_id).status
            print(status)
        if status == "failed":
            raise Exception(f"Batch job failed, ID is {batch_id}")
        elif status == "completed":
            print(f"Batch job {batch_id} completed successfully, sending next chunk")
        else: 
            print(f"Batch job {batch_id} status is {status}")
            print("Something is off")
            break

    time.sleep(60) # sleep for 1 minute before sending the next chunk


def run_chunk(chunk_n, client, filename_to_id, chunk_path, log_file_path):
  current_file_name = f"chunk_{chunk_n}.jsonl"
  new_file_name = f"chunk_{chunk_n}_sent.jsonl"
  file_id = filename_to_id[current_file_name]

  # Create a batch job
  batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
  )

  print(f"Created batch job with ID: {batch.id}")
  status = client.batches.retrieve(batch.id).status
  while status == "validating":
    time.sleep(5)
    status = client.batches.retrieve(batch.id).status
    print(status)

  if status == "failed":
      raise Exception(f"Batch job failed, ID is {batch.id}")
  else: 
    os.rename(os.path.join(chunk_path, current_file_name), os.path.join(chunk_path, new_file_name))
    print(f"Renamed {current_file_name} to {new_file_name}")
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%A, %B %d, %Y - %H:%M:%S')
    print(f"Sent {current_file_name} to OpenAI, batch id is {batch.id}, date is {formatted_datetime}")
    # save it to th log file passed as arg
    log_file = open(log_file_path, "a")
    log_file.write(f"\nSuccessfully sent {current_file_name} to OpenAI, batch id is {batch.id}, date is {formatted_datetime}\n")
    log_file.close()
    return batch.id
  
def print_status(client, id_to_filename):
    # prints the status of the batches
    for batch in client.batches.list():
        batch_id = batch.id
        errors = batch.errors
        input_file_id = batch.input_file_id
        output_file_id = batch.output_file_id
        status = batch.status
        if input_file_id in id_to_filename and output_file_id:
            input_file_name = id_to_filename[input_file_id]
            print(f"Batch {batch_id} is successful, status: {status}")
            print(f"Input file: {input_file_name}")
            print(f"Output file id: {output_file_id}")
            print("\n")
        elif input_file_id in id_to_filename and not output_file_id:
            input_file_name = id_to_filename[input_file_id]
            print(f"Batch {batch_id} has no output file - probably it is still running")
            print(f"Input file: {input_file_name}")
            print(f"Status: {status}")
            print("\n")

#
def download_results(client, output_folder_path, id_to_filename):
    # saves the outputs

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    # if in output_folder_path there are already files called like all values of id_to_filename, return
    if all([os.path.exists(os.path.join(output_folder_path, filename)) for filename in id_to_filename.values()]):
        print("All files have already been downloaded")
        return
    

    # total batches status
    counter = 0

    for batch in client.batches.list():
        counter += 1
        batch_id = batch.id
        errors = batch.errors
        input_file_id = batch.input_file_id
        output_file_id = batch.output_file_id
        status = batch.status
        if input_file_id in id_to_filename and output_file_id:
            input_file_name = id_to_filename[input_file_id]
            content = client.files.content(output_file_id)
            file_data_bytes = content.read()
            output_path = os.path.join(output_folder_path, id_to_filename[input_file_id])
            with open(output_path, "wb") as file:
                file.write(file_data_bytes)
            print(f"Output file saved to {output_path}")
        elif input_file_id in id_to_filename and not output_file_id:
            input_file_name = id_to_filename[input_file_id]
            print(f"Batch {batch_id} has no output file - probably it is still running")



    print(f"Total number of batches: {counter}")


def process_chunks(folder_path):
    """
        takes as input the folder containing the output files from the batch job in OpenAI, outputs a unique file
    
    """
    output_files = os.listdir(folder_path)
    final_output_file = os.path.join(folder_path, "final_output.jsonl")

    if os.path.exists(final_output_file): 
        print("Output already exists, exiting")
        return


    for current_file_name in output_files:
        with open(os.path.join(folder_path, current_file_name), 'r') as infile, open(final_output_file, 'a') as outfile:
            for line in infile:
                data = json.loads(line)
                custom_id = data["custom_id"]

                # Split the string by underscores
                split_string = custom_id.split("_")

                # Extract the query ID and source
                query_id = split_string[3]
                source = split_string[5]
                raw_answer = data["response"]["body"]["choices"][0]["message"]["content"]
                processed_answer = process_output_CoT(raw_answer)
                output_data = {
                    "query_id": query_id,
                    "source": source,
                    "is_causal_raw": raw_answer,
                    "is_causal": processed_answer
                }
                json.dump(output_data, outfile)
                outfile.write('\n')
            print(f"Processed {current_file_name}")
    
    print(f"Final output file saved to {final_output_file}")



if __name__ == "__main__":
    
    # parse arguments for cat and prompt function
    import argparse
    parser = argparse.ArgumentParser(description='Run the cat batch API')
    parser.add_argument('-cat', type=str, help='The cat category to be filled')
    parser.add_argument('-prompt_function', type=str, help='The prompt function to be used')
    parser.add_argument('-model', type=str, help='The model to be used')
    parser.add_argument('-input', type=str, help='The input file to be used')
    args = parser.parse_args()
    cat, prompt_function_name, model, input_file = args.cat, args.prompt_function, args.model, args.input

    prompt_function = globals()[prompt_function_name]

    output_folder = f"openai_batch_job_{cat}_{model}_{prompt_function_name}"
    # create a folder 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    formatted_file = convert_to_openai_input(input_file, output_folder, cat, prompt_function, model)

    chunk_path = chunkify(formatted_file, output_folder, 150000, model)

    client = OpenAI()
    filename_to_id, id_to_filename = files_upload_to_openai(chunk_path, client)

    send_requests_to_openai(client, filename_to_id, chunk_path)
    received_outputs_folder = os.path.join(output_folder, "received_outputs")
    print_status(client, id_to_filename)
    download_results(client, received_outputs_folder, id_to_filename)
    process_chunks(received_outputs_folder)