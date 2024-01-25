import argparse
import logging
import time
import uuid
import torch
import json
import re
import ast
import xml.etree.ElementTree as ET

from tqdm import tqdm
from tokenization_arcade100k import Arcade100kTokenizer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

class ModelEvaluator:
    def __init__(self, model_path, dpo="False"):
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            return_dict=True,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.tokenizer = Arcade100kTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.eval_results = []
        if dpo == "True":
            self.dpo_results = []
        print(self.model.config)
        print(self.model.generation_config)
        print(self.model.parameters)

    def validate_and_extract_tool_calls(self, completion, chat_template):
        # Define a pattern to find the assistant message
        if chat_template == "zephyr":
            assistant_pattern = re.compile(r'<\|assistant\|>((?:(?!<\|assistant\|>).)*)$', re.DOTALL)
        elif chat_template == "chatml":
            assistant_pattern = re.compile(r'<\\|im_start\\|>assistant((?:(?!<\\|im_start\\|>assistant).)*)$', re.DOTALL)
        assistant_match = assistant_pattern.search(completion)

        validation_result = False
        tool_calls = []

        if assistant_match:
            assistant_content = assistant_match.group(1).strip()

            xml_sections = re.split(r"(?<=</tool_call>)", assistant_content)
            for xml_section in xml_sections:
                if "<tool_call>" not in xml_section:
                    # Skip sections without opening tag
                    continue
                elif "</tool_call>" not in xml_section:
                    xml_section += "</tool_call>"

                try:
                    # Wrap section in root element
                    xml_section = f"<root>{xml_section}</root>"
                    # Parse XML
                    root = ET.fromstring(xml_section)

                    # Extract JSON data
                    for element in root.findall(".//tool_call"):
                        json_text = element.text.strip()

                        try:
                            # Prioritize json.loads for better error handling
                            json_data = json.loads(json_text)
                        except json.JSONDecodeError:
                            try:
                                # Fallback to ast.literal_eval if json.loads fails
                                json_data = ast.literal_eval(json_text)
                            except SyntaxError as err:
                                print(f"JSON parsing failed with both json.loads and ast.literal_eval:")
                                print(f"- JSON Decode Error: {err}")
                                print(f"- Problematic JSON text: {json_text}")
                                continue  # Skip to the next tool_call_element

                        tool_calls.append(json_data)
                        validation_result = True

                except ET.ParseError as err:
                    if "mismatched tag" in str(err):
                        # Skip section on mismatched tag error
                        print(err)
                    else:
                        print(f"XML Parse Error: {err}")

        else:
            print("No match found for the assistant pattern.")

        # Return default values if no valid data is extracted
        return validation_result, tool_calls, assistant_content
        
    def validate_func_calls(self, generated_arguments, expected_arguments):
        for key, expected_value in expected_arguments.items():
            if generated_arguments.get(key) != expected_value:
                print(f"Function args do not match; expected:{expected_value}\ngot:{generated_arguments.get(key)}")
                return "failed"
        return "passed"

    def evaluate_dataset(self, eval_dataset, chat_template, example="False"):

        for sample in tqdm(eval_dataset, desc="processing samples", unit="sample"):
            
            example_prompt = "###Example\nAn example usage of functions is as follows\n```\nSYSTEM: You are a helpful assistant who has access to functions. Use them if required\n<tools>[\n {\n \"name\": \"calculate_distance\",\n \"description\": \"Calculate the distance between two locations\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"origin\": {\n \"type\": \"string\",\n \"description\": \"The starting location\"\n },\n \"destination\": {\n \"type\": \"string\",\n \"description\": \"The destination location\"\n },\n \"mode\": {\n \"type\": \"string\",\n \"description\": \"The mode of transportation\"\n }\n },\n \"required\": [\n \"origin\",\n \"destination\",\n \"mode\"\n ]\n }\n },\n {\n \"name\": \"generate_password\",\n \"description\": \"Generate a random password\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"length\": {\n \"type\": \"integer\",\n \"description\": \"The length of the password\"\n }\n },\n \"required\": [\n \"length\"\n ]\n }\n }\n]\n\n</tools>\nUSER: Hi, I need to know the distance from New York to Los Angeles by car.\nASSISTANT:\n<tool_call>\n{\"arguments\": {\"origin\": \"New York\",\n \"destination\": \"Los Angeles\", \"mode\": \"car\"}, \"name\": \"calculate_distance\"}\n</tool_call>\n```\n"
            if example == "True":
                sample['prompt'][0]['content'] += example_prompt
                print(sample['prompt'][0])
            #prompt = [
            #    {'role': 'system', 'content': sample["system"]},
            #    {'role': 'user', 'content': sample["user"]}
            #]
            print(sample['prompt'])
            inputs = self.tokenizer.apply_chat_template(
                sample['prompt'],
                add_generation_prompt=True,
                return_tensors='pt'
            )

            tokens = self.model.generate(
                inputs.to(self.model.device),
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )

            completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False)
            print(completion)
            validation, tool_calls, assistant_message = self.validate_and_extract_tool_calls(completion, chat_template)
            print(assistant_message)

            sample['model_completion'] = ""
            sample['result'] = "failed"

            if validation:           
                eval_tool_calls = json.loads(sample['completion'])
                
                all_valid = True
                
                for eval_tool_call in eval_tool_calls:
                    function_found = False
                    
                    for tool_call in tool_calls:
                        if tool_call['name'] == eval_tool_call['name']:
                            function_found = True
                            result = self.validate_func_calls(tool_call['arguments'], eval_tool_call['arguments'])
                            sample['model_completion'] += f"<tool_call>\n{tool_call}\n</tool_call>\n"
                            print(f"{tool_call['name']} validation: {result}")
                            if result == "failed":
                                all_valid = False
                                break

                    if not function_found:
                        print(f"Function '{eval_tool_call['name']}' not found")
                        all_valid = False  
            else:
                print("Function call validation failed")
                sample['model_completion'] = assistant_message 
                all_valid = False
            
            if all_valid:
                sample['result'] = "passed"
            print(f"all validations: {sample['result']}")

            self.eval_results.append(sample)
            if self.dpo_results is not None and sample['result'] == "failed":
                chosen_completion = ""
                for tool_call in json.loads(sample['completion']):
                    chosen_completion += f"<tool_call>\n{tool_call}\n<tool_call>\n"
                self.dpo_results.append({
                            #"id": str(uuid.uuid4()),
                            "system": sample['system'],
                            "question": sample['user'],
                            "chosen": chosen_completion,
                            "rejected": sample['model_completion']
                })

    def calculate_pass_rate(self):
        passed_count =sum(1 for sample in self.eval_results if sample["result"] == "passed")
        pass_rate = passed_count / len(self.eval_results)
        return pass_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on fireworks-ai dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--example", type=str, default="False", help="Option to include one-shot example in sys prompt")
    parser.add_argument("--dpo", type=str, default="False", help="Option to create dpo sample")
    parser.add_argument("--num_samples", type=int, default=None, help="Option to subset eval dataset")
    args = parser.parse_args()
    
    # Load evaluation dataset
    #eval_dataset = load_dataset("NousResearch/func-calling-eval")['train']
    if args.num_samples:
        eval_dataset = load_dataset("interstellarninja/tool-calls-sampled-prompts", split=f'train[:{args.num_samples}]')
    else:
        eval_dataset = load_dataset("interstellarninja/tool-calls-sampled-prompts")['train']

    # Create model evaluator instance
    model_evaluator = ModelEvaluator(args.model_path, args.dpo)

    # Evaluate the dataset
    model_evaluator.evaluate_dataset(eval_dataset, args.chat_template, args.example)
    #results_path = '/home/interstellarninja/ai_projects/axolotl/examples/phi/eval_results.json'
    results_path = '/home/interstellarninja/ai_projects/axolotl/examples/stablelm/eval_results.json'
    with open(results_path, 'w') as file:
        json.dump(model_evaluator.eval_results, file)

    if args.dpo == "True":
        dpo_path = '/home/interstellarninja/ai_projects/axolotl/examples/stablelm/dpo_selfgen.json'
        with open(dpo_path, 'w') as file:
            json.dump(model_evaluator.dpo_results, file)

    # Calculate and print pass rate
    pass_rate = model_evaluator.calculate_pass_rate()
    print(f"fireworks-ai function-calling eval (pass@1): {pass_rate}")
