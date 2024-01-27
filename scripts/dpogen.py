import argparse
from datasets import Dataset
import utils
import uuid
import json
import os
from aiutilities import AIUtilities
from validator import FunctionSignature
from validator import validate_function_calls

class DPOGenerator:
    def __init__(self, folder_path, dpo_path, json_path, hub_dataset_path):
        self.folder_path = folder_path
        self.dpo_path = dpo_path
        self.json_path = json_path
        self.hub_dataset_path = hub_dataset_path
        self.ai_utilities = AIUtilities()
    
    def prepare_dpo_dataset(self, max_files=50):
        file_counter = 0
        dataset = []
        for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    # Assuming task files have the format "task.json"
                    if file.endswith("json"):
                        # Extract category, subcategory, and task from the file path
                        category, subcategory = os.path.relpath(root, self.folder_path).split(os.path.sep)[:2]
                        dpo_results_path = f"{self.folder_path}_dpo"
                        dpo_results_path =os.path.join(dpo_results_path, category, subcategory)
                        dpo_file_path = os.path.join(dpo_results_path, file)
                        
                        task, _ = os.path.splitext(file)
                        task = task.replace("_", " ")
                        
                        # Remove underscores from category and subcategory
                        category = category.replace("_", " ")
                        subcategory = subcategory.replace("_", " ")
                        
                        # Construct the full file path
                        file_path = os.path.join(root, file)
 
                        # Now you can do further processing with the file_path
                        #print(f"File Path: {file_paths}")
                        if not os.path.exists(dpo_file_path):
                            with open(file_path) as json_file:
                                json_data = json.load(json_file)
                            results = self.run_dpo_generation(json_data)
                            if results:
                                dataset.append(results)
                                os.makedirs(dpo_results_path, exist_ok=True)
                                with open(dpo_file_path, 'w') as dpo_file:
                                    json.dump(results, dpo_file, indent=2)
                                file_counter += 1
                                if file_counter >= max_files:
                                    break
                            else:
                                continue
        return dataset
    
    def run_tool_correction(self, user_message, tools, results):
        prompt = user_message
        prompt += f"\n Here's the previously called functions with error message that need to be called correctly: {results}"
        func_call_messages = [
            {"role": "user", "content": prompt}
        ]
        tool_call_response = self.ai_utilities.run_ai_tool_completion(func_call_messages, tools, tool_choice="auto")
        tool_call_message = {key: value for key,  value in tool_call_response.model_dump().items() if value is not None}
        print(tool_call_message)
        if "tool_calls" in tool_call_message and tool_call_message["tool_calls"]:
            tool_calls = tool_call_message["tool_calls"]
        return tool_calls

    def run_validation_loop(self, tool_calls, user_message, function_signatures):
        accepted = None
        rejected = None
        results, failed_flag = validate_function_calls(tool_calls, function_signatures)
        if failed_flag:
            rejected = tool_calls
            retry_count = 0
            while failed_flag and retry_count < 3:
                tool_calls = self.run_tool_correction(user_message, tool_calls, results)        
                results, failed_flag = validate_function_calls(tool_calls, function_signatures)
                retry_count += 1
            if not failed_flag:
                accepted = tool_calls
            return accepted, rejected
        return accepted, rejected

    def run_dpo_generation(self, conversation):
        messages = conversation["messages"]
        tools = conversation["tools"]
        tool_calls = None

        for message in messages:
            if message["role"] == "user":
                user_message = message.get("content", "")

            if message["role"] == "assistant":
                if "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message["tool_calls"]

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

        if tool_calls:
            accepted, rejected = self.run_validation_loop(tool_calls, user_message, function_signatures)

            if accepted and rejected:
                accepted_calls = ""
                for tool_call in accepted:

                    accepted_calls += f"```tool_call\n{tool_call['function']}\n```\n"
                rejected_calls = ""
                for tool_call in rejected:
                    rejected_calls += f"```tool_call\n{tool_call['function']}\n```\n"
                return {"id": str(uuid.uuid4()),
                        "system": f'```tools\n{tools}\n```',
                        "user": user_message,
                        "chosen": accepted_calls,
                        "rejected": rejected_calls}         
        return None
    
    def load_dpo_dataset(self):
        dpo_dataset = []
        for root, dirs, files in os.walk(self.dpo_path):
            for file in files:
                # Assuming task files have the format "task.json"
                if file.endswith("json"):
                    # Extract category, subcategory, and task from the file path
                    category, subcategory = os.path.relpath(root, self.dpo_path).split(os.path.sep)[:2]

                    task, _ = os.path.splitext(file)
                    task = task.replace("_", " ")
                    
                    # Remove underscores from category and subcategory
                    category = category.replace("_", " ")
                    subcategory = subcategory.replace("_", " ")
                    
                    # Construct the full file path
                    file_path = os.path.join(root, file)
                    
                    # Now you can do further processing with the file_path
                    print(f"File Path: {file_path}")
                    with open(file_path) as file:
                        json_data = json.load(file)
                    
                     # prepare system message with function signatures
                    id_value = json_data.pop("id")
                    sys_prompt = "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions."
                    sys_prompt += f'\n<tools>\n{utils.extract_tool_code_block(json_data["system"])}\n<tools>\n'
                    sys_prompt += "For each function call return a json object with function name and arguments within <tool_call> </tool_call> XML tags with the following schema:\n<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>\n"
                    user_message = json_data.pop("human")
                    
                    chosen_content = utils.extract_toolcall_code_blocks(json_data.pop("accepted"))
                    
                    if chosen_content:
                        chosen_message = ""
                        for tool_call in chosen_content:
                            chosen_message += f"<tool_call>\n{tool_call}\n</tool_call>\n"
                    else:
                        chosen_message = json_data.pop("accepted")

                    rejected_content = utils.extract_toolcall_code_blocks(json_data.pop("rejected"))
                    
                    if rejected_content:
                        rejected_message = ""
                        for tool_call in rejected_content:
                            rejected_message += f"<tool_call>\n{tool_call}\n</tool_call>\n"
                    else:
                        rejected_message = json_data.pop("rejected")

                    print(f"accepted: {chosen_message}\nrejected: {rejected_message}")

                    dpo_dataset.append({
                        #"id": id_value,
                        "system": sys_prompt,
                        "question": user_message,
                        "chosen": chosen_message,
                        "rejected": rejected_message
                        #"category": category,
                        #"subcategory": subcategory,
                        #"task": task
                    })

        return dpo_dataset
                    

    def format_and_upload_to_hub(self, upload=False):
        dpo_dataset = self.load_dpo_dataset()     
        if os.path.exists("./dpo_selfgen.json"):
            with open("./dpo_selfgen.json") as file:
                dpo_selfgen = json.load(file)
            dpo_dataset += dpo_selfgen
        if os.path.exists("./fireworks_dpo.json"):
            with open("./fireworks_dpo.json") as file:
                fireworks_dpo = json.load(file)
            dpo_dataset += fireworks_dpo    
        dataset = Dataset.from_list(dpo_dataset)
        with open(self.json_path, 'w') as file:
            json.dump(dpo_dataset, file)

        if upload:
            dataset.push_to_hub(
                self.hub_dataset_path,
                #commit_message="Upload ShareGPT-formatted dataset"
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO dataset")
    parser.add_argument("--max_files", type=int, default=50, help="Maximum number of files to process")
    parser.add_argument("--run_type", type=str, default="generator", help="type of run")

    args = parser.parse_args()

     # Load configuration from YAML file
    config_path = "./config.yaml"
    config = utils.load_yaml(config_path)

    # get paths from config file
    results_path = config["paths"]["results_corrected"]
    dpo_path = config["paths"]["dpo_path"]
    json_path = config["paths"]["json_path"]
    hf_dataset_path = config["paths"]["hf_dpo_path"]

    generator = DPOGenerator(results_path, dpo_path, json_path, hf_dataset_path)
    if args.run_type == "generator":
        generator.prepare_dpo_dataset(max_files=args.max_files)
    if args.run_type == "uploader":
        generator.format_and_upload_to_hub(upload=True)
