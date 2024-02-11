import argparse
import ast
from datasets import Dataset
from pydantic import BaseModel, ValidationError, create_model

import utils
import uuid
import json
import os
from validator import FunctionSignature
from validator import validate_function_calls

class ShareGPTDatasetUploader:
    def __init__(self, folder_path, output_path, hub_dataset_path):
        self.folder_path = folder_path
        self.output_path = output_path
        self.hub_dataset_path = hub_dataset_path

    def prepare_sharegpt_dataset(self, turn):
        output_data = []

        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                # Assuming task files have the format "task.json"
                if file.endswith("json"):
                    # Extract category, subcategory, and task from the file path
                    category, subcategory = os.path.relpath(root, self.folder_path).split(os.path.sep)[:2]

                    task, _ = os.path.splitext(file)
                    task = task.replace("_", " ")
                    
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
                        output_data.append({"id": unique_id, "conversations": converted_conversation, "category": category, "subcategory": subcategory, "task": task})
        return output_data

    def validate_json_object(self, data, schema):

        # Pre-process data
        data = json.loads(json.dumps(data))

        # Pre-process schema
        schema = json.loads(json.dumps(schema))
        
        # Create Pydantic models and validate
        schema_model = self.create_model(schema)
        try: 
            schema_model(**data)
        except ValidationError as e:
            print(f"Validation Error: {e}")
            return False

        return True

    def create_model(self, schema_dict, base_name="Model"):
        fields = {}
        for name, prop in schema_dict.items():
            if "$ref" in prop:
                ref_model_name = prop["$ref"].split("/")[-1]
                fields[name] = (create_model(schema_dict[ref_model_name], base_name=ref_model_name),)
            else:
                fields[name] = (prop["type"],)
        
        model_name = f"{base_name}Model"
        model = type(model_name, (BaseModel,), fields)
        return model

    def convert_to_sharegpt(self, conversation, turn="multi"):
        converted_conversation = []
        failed_flag = False

        # Load JSON schema
        schema_dict = conversation["pydantic_schema"]

        # prepare system message with function signatures
        sys_prompt = "You are a helpful assistant that answers in JSON. You are provided with pydantic schema within <schema> </schema> XML tags."
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
            is_valid = self.validate_json_object(json_object, schema_dict)
            if not is_valid:
                failed_flag = False
        except Exception as e:
            failed_flag = True
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

        if upload == "True":
            dataset.push_to_hub(
                self.hub_dataset_path,
                #commit_message="Upload ShareGPT-formatted dataset"
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and Uplaod ShareGPT dataset")
    parser.add_argument("--turn", type=str, default="multi", help="type of turn")
    parser.add_argument("--upload", type=str, default="False", help="whether to upload to hub or just save locally")

    args = parser.parse_args()

    # Convert upload argument to boolean
    upload = ast.literal_eval(args.upload)

     # Load configuration from YAML file
    config_path = "./config.yaml"
    config = utils.load_yaml(config_path)

    # get paths from config file
    results_path = config["paths"]["results_corrected"]
    dataset_path = config["paths"]["json_path"]
    hf_dataset_path = config["paths"]["hf_dataset_path"]

    uploader = ShareGPTDatasetUploader(results_path, dataset_path, hf_dataset_path)
    uploader.format_and_upload_to_hub(turn=args.turn, upload=upload)
