from datasets import Dataset
import utils
import uuid
import json
import os

class ShareGPTDatasetUploader:
    def __init__(self, folder_path, output_path, hub_dataset_path):
        self.folder_path = folder_path
        self.output_path = output_path
        self.hub_dataset_path = hub_dataset_path

    def prepare_sharegpt_dataset(self):
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
                    converted_conversation = self.convert_to_sharegpt(json_data)
                    output_data.append({"id": unique_id, "conversations": converted_conversation, "category": category, "subcategory": subcategory, "task": task})
        return output_data

    def convert_to_sharegpt(self, conversation):
        converted_conversation = []
        tool_results = None
        summary_message = None

        # prepare system message with function signatures
        system_message = {
            "from": "system",
            "value": f'```tools\n{conversation["tools"]}\n```'
        }
        converted_conversation.append(system_message)

        tool_results = '```tool_response\n'
        for message in conversation["messages"]:
            role = message["role"]
            content = message.get("content", "")

            if role == "user":
                user_message = {"from": "human", "value": content}
                converted_conversation.append(user_message)
            elif role == "assistant":
                if "tool_calls" in message and message["tool_calls"] is not None:
                    tool_calls = message["tool_calls"]
                    # concatenate multiple tool calls
                    gpt_value = ""
                    for tool_call in tool_calls:
                        gpt_value += f"```tool_call\n{tool_call['function']}\n```\n"
                    tool_call_message = {"from": "gpt", "value": gpt_value}
                    converted_conversation.append(tool_call_message)
                else:
                    summary_message = {"from": "gpt", "value": content}
            elif role == "tool":
                function_name = message["name"]
                tool_call_id = message["tool_call_id"]
                function_content = message["content"]
                combined_value = f'{{"name": "{function_name}", "content": {function_content}}}'
                
                # concatenate multiple tool call results 
                tool_results += f'{combined_value}\n'

        # Check if tool_message is present and append it
        if tool_results:
            tool_results += '```'
            tool_message = {"from": "tool", "value": tool_results}
            converted_conversation.append(tool_message)

        # Check if summary_message is present and append it
        if summary_message:
            converted_conversation.append(summary_message)
            
        return converted_conversation

    def format_and_upload_to_hub(self, upload=False):
        sharegpt_format_data = self.prepare_sharegpt_dataset()
        dataset = Dataset.from_list(sharegpt_format_data)

        with open(self.output_path, 'w') as file:
            json.dump(sharegpt_format_data, file)

        if upload:
            dataset.push_to_hub(
                self.hub_dataset_path,
                #commit_message="Upload ShareGPT-formatted dataset"
            )
if __name__ == "__main__":
     # Load configuration from YAML file
    config_path = "./config.yaml"
    config = utils.load_yaml(config_path)

    # get paths from config file
    results_path = config["paths"]["results_corrected"]
    dataset_path = config["paths"]["dataset_path"]
    hf_dataset_path = config["paths"]["hf_dataset_path"]

    uploader = ShareGPTDatasetUploader(results_path, dataset_path, hf_dataset_path)
    uploader.format_and_upload_to_hub(upload=True)
