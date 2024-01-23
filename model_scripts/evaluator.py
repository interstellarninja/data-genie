import argparse
import logging
import torch
import json
import re
import ast
import xml.etree.ElementTree as ET
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelEvaluator:
    def __init__(self, model_path):
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.eval_results = []

    def validate_and_extract_tool_calls(self, completion, chat_template):
        # Define a pattern to find the assistant message
        if chat_template == "zephyr":
            assistant_pattern = re.compile(r'<\|assistant\|>((?:(?!<\|assistant\|>|</s>).)*)</s>', re.DOTALL)
        elif chat_template == "chatml":
            assistant_pattern = re.compile(r'<\\|im_start\\|>assistant((?:(?!<\\|im_start\\|>assistant).)*)$', re.DOTALL)
        assistant_match = assistant_pattern.search(completion)

        validation_result = False
        extracted_data = []
        if assistant_match:
            assistant_content = assistant_match.group(1).strip()
            print(assistant_content)

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
                            json_data = json.loads(json_text)  # Use json.loads first
                        except json.JSONDecodeError:
                            try:
                                # Fallback to ast.literal_eval if json.loads fails
                                json_data = ast.literal_eval(json_text)
                            except SyntaxError as err:
                                print(f"JSON parsing failed with both json.loads and ast.literal_eval:")
                                print(f"- JSON Decode Error: {err}")
                                print(f"- Problematic JSON text: {json_text}")
                                validation_result = False
                                continue  # Skip to the next tool_call_element

                        extracted_data.append(json_data)
                        validation_result = True
                            
                except ET.ParseError as err:
                    if "mismatched tag" in str(err):
                        # Skip section on mismatched tag error
                        print(err)
                    else:
                        print(f"XML Parse Error: {err}")
        else:
            print("No match found for the assistant pattern.")
            return validation_result, extracted_data
        
    def validate_func_calls(self, generated_arguments, expected_arguments):
        for key, expected_value in expected_arguments.items():
            if generated_arguments.get(key) != expected_value:
                print(f"Function args do not match; expected:{expected_value}, got:{generated_arguments.get(key)}")
                return "failed"
        return "passed"

    def evaluate_dataset(self, eval_dataset, chat_template, example="False"):

        for sample in eval_dataset:
            example_prompt = "###Example\nAn example usage of functions is as follows\n```\nSYSTEM: You are a helpful assistant who has access to functions. Use them if required\n<tools>[\n {\n \"name\": \"calculate_distance\",\n \"description\": \"Calculate the distance between two locations\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"origin\": {\n \"type\": \"string\",\n \"description\": \"The starting location\"\n },\n \"destination\": {\n \"type\": \"string\",\n \"description\": \"The destination location\"\n },\n \"mode\": {\n \"type\": \"string\",\n \"description\": \"The mode of transportation\"\n }\n },\n \"required\": [\n \"origin\",\n \"destination\",\n \"mode\"\n ]\n }\n },\n {\n \"name\": \"generate_password\",\n \"description\": \"Generate a random password\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"length\": {\n \"type\": \"integer\",\n \"description\": \"The length of the password\"\n }\n },\n \"required\": [\n \"length\"\n ]\n }\n }\n]\n\n</tools>\nUSER: Hi, I need to know the distance from New York to Los Angeles by car.\nASSISTANT:\n<tool_call>\n{\"arguments\": {\"origin\": \"New York\",\n \"destination\": \"Los Angeles\", \"mode\": \"car\"}, \"name\": \"calculate_distance\"}\n</tool_call>\n```\n"
            if example == "True":
                sample['prompt'][0]['content'] += example_prompt
                print(sample['prompt'][0])
            #prompt = [
            #    {'role': 'system', 'content': sample["system"]},
            #    {'role': 'user', 'content': sample["user"]}
            #]
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

            validation, assistant_message = self.validate_and_extract_tool_calls(completion, chat_template)
            print(assistant_message)

            if validation:
                function_found = False
                eval_tool_calls = json.loads(sample['completion'])
                for tool_call in assistant_message:
                    if tool_call['name'] == eval_tool_calls['name']:
                        result = self.validate_func_calls(tool_call['arguments'], eval_tool_calls['arguments'])
                        print(result)
                        function_found = True
                        break

                if not function_found:
                    print("Function not found")
                    result = "failed"
                    print(result)
            else:
                print("function call validation failed")
                result = "failed"
                print(result)

            sample['model_completion'] = assistant_message
            sample['result'] = result

            self.eval_results.append(sample)
    
    def calculate_pass_rate(self):
        passed_count =sum(1 for sample in self.eval_results if sample["result"] == "passed")
        pass_rate = passed_count / len(self.eval_results)
        return pass_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on fireworks-ai dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--example", type=str, default="False", help="Option to include one-shot example in sys prompt")
    args = parser.parse_args()
    
    # Load evaluation dataset
    eval_dataset = load_dataset("NousResearch/func-calling-eval")['train']

    # Create model evaluator instance
    model_evaluator = ModelEvaluator(args.model_path)

    # Evaluate the dataset
    model_evaluator.evaluate_dataset(eval_dataset, args.chat_template, args.example)
    results_path = '/home/interstellarninja/ai_projects/axolotl/examples/phi/eval_results.json'
    with open(results_path, 'w') as file:
        json.dump(model_evaluator.eval_results, file)

    # Calculate and print pass rate
    pass_rate = model_evaluator.calculate_pass_rate()
    print(f"fireworks-ai function-calling eval (pass@1): {pass_rate}")
