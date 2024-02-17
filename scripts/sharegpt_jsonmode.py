import argparse
import ast
from datasets import Dataset
from pydantic import BaseModel, ValidationError, create_model
import utils
import uuid
import json
import os
from validator import FunctionSignature, validate_json_data
from validator import validate_json_object

class ShareGPTDatasetUploader:
    def __init__(self, folder_path, output_path, hub_dataset_path):
        self.folder_path = folder_path
        self.output_path = output_path
        self.hub_dataset_path = hub_dataset_path

    def prepare_sharegpt_dataset(self, turn):
        output_data = []
        print(self.folder_path)
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                # Assuming task files have the format "task.json"
                if file.endswith("json"):
                    # Extract category, subcategory, and task from the file path
                    category, subcategory = os.path.relpath(root, self.folder_path).split(os.path.sep)[:2]

                    task, _ = os.path.splitext(file)
                    task = task.replace("_", " ")

                    task = utils.strip_incomplete_text(task)
                    
                    # Remove underscores from category and subcategory
                    category = category.replace("_", " ")
                    subcategory = subcategory.replace("_", " ")
                    
                    unique_id = str(uuid.uuid4())
                    
                    # Construct the full file path
                    file_path = os.path.join(root, file)
                    
                    # Now you can do further processing with the file_path
                    print(f"File Path: {file_path}")
                    with open(file_path) as file:
                        json_data = json.load(file)
                    
                    converted_conversation = self.convert_to_sharegpt(json_data, turn)
                    if converted_conversation:
                        output_data.append({"id": unique_id, "conversations":json.dumps(converted_conversation), "category": category, "subcategory": subcategory, "task": task})
        return output_data

    def convert_to_sharegpt(self, conversation, turn="multi"):
        converted_conversation = []
        failed_flag = True

        # Load JSON schema
        schema_dict = conversation["pydantic_schema"]

        # prepare system message with function signatures
        sys_prompt = "You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:"
        sys_prompt += f'\n<schema>\n{schema_dict}\n</schema>\n'
        system_message = {
            "from": "system",
            "value": sys_prompt
        }
        converted_conversation.append(system_message)

        for message in conversation["messages"]:
            role = message["role"]
            content = message.get("content", "")

            if role == "user":
                user_message = {"from": "human", "value": content}
                 
            elif role == "assistant":
                json_object = content
                assistant_message = {"from": "assistant", "value": json.dumps(json_object)}
        try:
            is_valid = validate_json_data(json_object, schema_dict)
            if is_valid:
                print(is_valid)
                failed_flag = False
        except Exception as e:
            print(f"{e}")
        if not failed_flag:
            converted_conversation.append(user_message)
            if json_object:
                converted_conversation.append(assistant_message)
                print(converted_conversation)
                return converted_conversation    
        return None

    def format_and_upload_to_hub(self, turn, upload):
        sharegpt_format_data = self.prepare_sharegpt_dataset(turn)
       
        if os.path.exists("./extraction_data.json"):
            with open("./extraction_data.json") as file:
                extraction_data = json.load(file)
            sharegpt_format_data += extraction_data
        
        dataset = Dataset.from_list(sharegpt_format_data)

        with open(self.output_path, 'w') as file:
            json.dump(sharegpt_format_data, file)

        if upload:
            print('Dataset uploaded to hub')
            dataset.push_to_hub(
                self.hub_dataset_path,
                commit_message="Upload ShareGPT-formatted dataset"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and Uplaod ShareGPT dataset")
    parser.add_argument("--turn", type=str, default="multi", help="type of turn")
    parser.add_argument("--upload", type=bool, default=False, help="whether to upload to hub or just save locally")
    parser.add_argument("--results_path", type=str, help="path to generated results")

    args = parser.parse_args()

     # Load configuration from YAML file
    config_path = "./config.yaml"
    config = utils.load_yaml(config_path)

    # get paths from config file
    if args.results_path:
        results_path = args.results_path
        print(results_path)
    else:
        results_path = config["paths"]["results_corrected"]
    dataset_path = config["paths"]["json_path"]
    hf_dataset_path = config["paths"]["hf_dataset_path"]

    uploader = ShareGPTDatasetUploader(results_path, dataset_path, hf_dataset_path)
    uploader.format_and_upload_to_hub(turn=args.turn, upload=args.upload)
