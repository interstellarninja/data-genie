import argparse
import ast
from datasets import Dataset
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

    def convert_to_sharegpt(self, conversation, turn="multi"):
        converted_conversation = []
        tool_results = None
        tool_call_message = None
        summary_message = None
        failed_flag = False

        function_signature_dicts = conversation["tools"]

        function_signatures = []

        for signature_dict in function_signature_dicts:
            try:
                # Check if the required fields are present in the signature_dict
                if "function" in signature_dict and "name" in signature_dict["function"]:
                    function_signature = FunctionSignature(**signature_dict)
                    function_signatures.append(function_signature)
                else:
                    print(f"Missing required fields in function signature: {signature_dict}")
            except Exception as e:
                # Handle validation errors
                print(f"Validation error for function signature: {e}")

        # prepare system message with function signatures
        sys_prompt = "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions."
        sys_prompt += f'\n<tools>\n{conversation["tools"]}\n</tools>\n'
        sys_prompt += "For each function call return a json object with function name and arguments within <tool_call></tool_call> tags with the following schema:\n<tool_call>\n{'arguments': <args dict>, 'name': <function name>}\n</tool_call>\n"
        system_message = {
            "from": "system",
            "value": sys_prompt
        }
        converted_conversation.append(system_message)

        tool_results = ""
        for message in conversation["messages"]:
            role = message["role"]
            content = message.get("content", "")

            if role == "user":
                user_message = {"from": "human", "value": content}
                
            
            elif role == "assistant":
                if "tool_calls" in message and message["tool_calls"] is not None:
                    tool_calls = message["tool_calls"]
                    results, failed_flag = validate_function_calls(tool_calls, function_signatures)
                    # concatenate multiple tool calls
                    if not failed_flag:
                        gpt_value = ""
                        for tool_call in tool_calls:
                            gpt_value += f"<tool_call>\n{tool_call['function']}\n</tool_call>\n"
                        tool_call_message = {"from": "gpt", "value": gpt_value}
                    else:
                        print(results)                  
                else:
                    summary_message = {"from": "gpt", "value": content}
            
            elif role == "tool":
                function_name = message["name"]
                tool_call_id = message["tool_call_id"]
                function_content = message["content"]
                combined_value = f'{{"name": "{function_name}", "content": {function_content}}}'
                
                # concatenate multiple tool call results 
                tool_results += f"<tool_response>\n{combined_value}\n</tool_response>\n"
        
        if not failed_flag:
            converted_conversation.append(user_message)
            if tool_call_message:
                converted_conversation.append(tool_call_message)
            # Check if tool_message is present and append it
            if turn=="multi":
                if tool_results:
                    tool_message = {"from": "tool", "value": tool_results}
                    converted_conversation.append(tool_message)

                # Check if summary_message is present and append it
                if summary_message:
                    converted_conversation.append(summary_message)
                return converted_conversation
            elif turn=="single":
                if not tool_call_message:
                    converted_conversation.append(summary_message)
                return converted_conversation    
        return None

    def format_and_upload_to_hub(self, turn, upload=False):
        sharegpt_format_data = self.prepare_sharegpt_dataset(turn)
        dataset = Dataset.from_list(sharegpt_format_data)

        with open(self.output_path, 'w') as file:
            json.dump(sharegpt_format_data, file)

        if upload:
            dataset.push_to_hub(
                self.hub_dataset_path,
                #commit_message="Upload ShareGPT-formatted dataset"
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and Uplaod ShareGPT dataset")
    parser.add_argument("--turn", type=str, default="multi", help="type of turn")
    parser.add_argument("--upload", type=str, default="True", help="whether to upload to hub or just save locally")

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
