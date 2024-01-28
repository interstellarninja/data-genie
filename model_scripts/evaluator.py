import argparse
import logging
import time
import uuid
import torch
import json
import re
import ast
import pygments  
from pygments import formatters, lexers
import xml.etree.ElementTree as ET
from colorama import init, Fore, Back, Style

from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
# set up logging configuration
logging.basicConfig(filename='model_evaluation.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# Create a custom formatter for colorized logs
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        log_message = super(ColorFormatter, self).format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{Fore.RESET}"
        return log_message

console_color_formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the colorized formatter to the console handler
console_handler.setFormatter(console_color_formatter)

console_json_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - \n%(message)s')
console_xml_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - \n%(message)s')

class ModelEvaluator:
    def __init__(self, model_path, dpo="False"):
        self.logger = logging.getLogger(self.__class__.__name__)
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.eval_results = []
        if dpo == "True":
            self.dpo_results = []
        self.logger.info(self.model.config)
        self.logger.info(self.model.generation_config)
        self.logger.info(self.model.parameters)
        self.logger.info(self.tokenizer.chat_template)
        self.logger.info(self.tokenizer.special_tokens_map)

    def validate_and_extract_tool_calls(self, completion, chat_template):
        # Define a pattern to find the assistant message
        if chat_template == "zephyr":
            assistant_pattern = re.compile(r'<\|assistant\|>((?:(?!<\|assistant\|>).)*)$', re.DOTALL)
        elif chat_template == "chatml":
            assistant_pattern = re.compile(r'<\\|im_start\\|>\s*assistant((?:(?!<\\|im_start\\|>\s*assistant).)*)$', re.DOTALL)
            #assistant_pattern = re.compile(r'<\\|im_start\\|>assistant((?:(?!<\\|im_start\\|>assistant).)*)$', re.DOTALL)
        assistant_match = assistant_pattern.search(completion)

        validation_result = False
        tool_calls = []
        
        if assistant_match:
            assistant_content = assistant_match.group(1).strip()
            #assistant_content = assistant_content.split("<|im_end|>")[0]

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
                                self.logger.info("%sJSON parsing failed with both json.loads and ast.literal_eval:%s", Fore.RED, Fore.RESET)
                                self.logger.info("%s- JSON Decode Error: %s%s", Fore.RED, err, Fore.RESET)
                                self.logger.info("%s- Problematic JSON text: %s%s", Fore.RED, json_text, Fore.RESET)

                        tool_calls.append(json_data)
                        validation_result = True

                except ET.ParseError as err:
                    if "mismatched tag" in str(err):
                        # Skip section on mismatched tag error
                        self.logger.info(Fore.YELLOW + err + Fore.RESET)
                    else:
                        self.logger.info("%sXML Parse Error: %s%s", Fore.RED, err, Fore.RESET)

        else:
            assistant_content = ""
            self.logger.info("%sNo match found for the assistant pattern.%s", Fore.RED, Fore.RESET)

        # Return default values if no valid data is extracted
        return validation_result, tool_calls, assistant_content
        
    def validate_func_calls(self, generated_arguments, expected_arguments):
        for key, expected_value in expected_arguments.items():
            if generated_arguments.get(key) != expected_value:
                self.logger.info("%sExpected %s: %s", Fore.GREEN, key, expected_value)
                self.logger.info("%sGot: %s%s", Fore.RED, generated_arguments.get(key), Style.RESET_ALL)
                self.logger.info(Style.RESET_ALL)
                return "failed"
        return "passed"

    def print_validation_message(self, expected, got, arg_name):
        # ANSI escape codes for styling
        bold = "\033[1m"
        red = "\033[91m"
        end_color = "\033[0m"

        # Format the error message
        error_message = (
            f"{bold}Function args do not match;{end_color} "
            f"{red}expected:{expected}{end_color} "
            f"{red}got:{got}{end_color}\n"
            f"{bold}{arg_name} validation:{end_color} {red}failed{end_color}"
        )
        self.logger.info(error_message)

    def highlight_syntax(self, content, format="json"):
        formatter = formatters.TerminalFormatter()

        if format == "json":
            lexer = lexers.get_lexer_by_name("json")
        elif format == "xml":
            lexer = lexers.get_lexer_by_name("xml")
        else:
            print("Unknown highlight format")
            return content

        highlighted = pygments.highlight(content, lexer, formatter)
        return highlighted

    def evaluate_model(self, eval_dataset, chat_template, example="False"):

        for sample in tqdm(eval_dataset, desc=Fore.BLUE+"processing samples"+Fore.RESET, unit="sample"):  
            example_prompt = "###Example\nAn example usage of functions is as follows\n```\nSYSTEM: You are a helpful assistant who has access to functions. Use them if required\n<tools>[\n {\n \"name\": \"calculate_distance\",\n \"description\": \"Calculate the distance between two locations\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"origin\": {\n \"type\": \"string\",\n \"description\": \"The starting location\"\n },\n \"destination\": {\n \"type\": \"string\",\n \"description\": \"The destination location\"\n },\n \"mode\": {\n \"type\": \"string\",\n \"description\": \"The mode of transportation\"\n }\n },\n \"required\": [\n \"origin\",\n \"destination\",\n \"mode\"\n ]\n }\n },\n {\n \"name\": \"generate_password\",\n \"description\": \"Generate a random password\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"length\": {\n \"type\": \"integer\",\n \"description\": \"The length of the password\"\n }\n },\n \"required\": [\n \"length\"\n ]\n }\n }\n]\n\n</tools>\nUSER: Hi, I need to know the distance from New York to Los Angeles by car.\nASSISTANT:\n<tool_call>\n{\"arguments\": {\"origin\": \"New York\",\n \"destination\": \"Los Angeles\", \"mode\": \"car\"}, \"name\": \"calculate_distance\"}\n</tool_call>\n```\n"
            if example == "True":
                sample['prompt'][0]['content'] += example_prompt
                self.logger.info(self.highlight_syntax(json.dumps(sample['prompt']), "json"))
            #prompt = [
            #    {'role': 'system', 'content': sample["system"]},
            #    {'role': 'user', 'content': sample["user"]}
            #]
            #print(self.highlight_syntax(json.dumps(sample['prompt']), "json"))
            
            if chat_template == "chatml" and self.tokenizer.chat_template is not None:
                self.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

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
            #self.logger.info(Fore.MAGENTA+f"model completion with eval prompt:\n{completion}"+Fore.RESET)
            self.logger.info(f"model completion with eval prompt:\n{completion}")
            validation, tool_calls, assistant_message = self.validate_and_extract_tool_calls(completion, chat_template)
            #print(Fore.GREEN+assistant_message+Fore.RESET)

            sample['model_completion'] = ""
            sample['result'] = "failed"

            if validation:
                if not hasattr(self, 'dpo_results'):        
                    eval_tool_calls = [json.loads(sample['completion'])]
                else:
                    eval_tool_calls = json.loads(sample['completion'])
                
                all_valid = True        
                for eval_tool_call in eval_tool_calls:
                    function_found = False
                    for tool_call in tool_calls:
                        if "name" not in tool_call:
                            self.logger.info(Fore.RED + f"Error: Tool call does not contain required 'name' key: {tool_call}" + Fore.RESET)
                            all_valid = False
                            continue
                            
                        if tool_call['name'] == eval_tool_call['name']:
                            function_found = True
                            if "arguments" not in tool_call:
                                self.logger.info(Fore.RED + f"Error: Tool call {tool_call['name']} does not contain required 'arguments' key" + Fore.RESET)
                                all_valid = False
                                continue

                            result = self.validate_func_calls(tool_call['arguments'], eval_tool_call['arguments'])
                            sample['model_completion'] += f"<tool_call>\n{tool_call}\n</tool_call>\n"
                            #self.logger.info(Fore.YELLOW+f"{tool_call['name']} validation: {result}"+Fore.RESET)
                            self.logger.info(f"{tool_call['name']} validation: {result}")
                            if result == "failed":
                                all_valid = False
                                break
                                
                    if not function_found:
                        self.logger.info(Fore.RED+f"Function '{eval_tool_call['name']}' not found"+Fore.RESET) 
                        all_valid = False 
            else:
                self.logger.info("%sFunction call validation failed%s", Fore.RED, Fore.RESET)
                sample['model_completion'] = assistant_message 
                all_valid = False
            
            if all_valid:
                sample['result'] = "passed"
                #self.logger.info(Fore.GREEN+f"all validations: {sample['result']}"+Fore.RESET)
                self.logger.info(f"all validations: {sample['result']}")
                self.logger.info(console_json_formatter.format(logging.makeLogRecord({'msg': json.dumps(tool_calls, indent=2)})))
                #print(f"passed tool calls:\n{self.highlight_syntax(json.dumps(tool_calls, indent=2), 'json')}")
            else:
                self.logger.info(Fore.RED+f"all validations: {sample['result']}"+Fore.RESET)
                self.logger.info(f"all validations: {sample['result']}")
                self.logger.info(console_xml_formatter.format(logging.makeLogRecord({'msg': assistant_message})))
                #print(f"failed assistant message\n{self.highlight_syntax(assistant_message, 'xml')}")

            self.eval_results.append(sample)
            if hasattr(self, 'dpo_results') and sample['result'] == "failed":
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
        self.logger.info("%sNumber of eval tests passed:%s%d", Fore.GREEN, Fore.RESET, passed_count)
        self.logger.info("%sNumber of eval tests failed:%d", Fore.RED, len(self.eval_results) - passed_count)
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

    # Initialize colorama
    init(autoreset=True)

    # Load evaluation dataset
    if args.dpo == "False":
        if args.num_samples:
            eval_dataset = load_dataset("NousResearch/func-calling-eval", split=f'train[:{args.num_samples}]')
        else:
            eval_dataset = load_dataset("NousResearch/func-calling-eval")['train']
    elif args.dpo == "True":
        if args.num_samples:
            eval_dataset = load_dataset("interstellarninja/tool-calls-sampled-prompts", split=f'train[:{args.num_samples}]')
        else:
            eval_dataset = load_dataset("interstellarninja/tool-calls-sampled-prompts")['train']

    # Create model evaluator instance
    model_evaluator = ModelEvaluator(args.model_path, args.dpo)

    # Evaluate the dataset
    model_evaluator.evaluate_model(eval_dataset, args.chat_template, args.example)

    # Calculate and print pass rate
    pass_rate = model_evaluator.calculate_pass_rate()
    if args.dpo == "False":
        print(f"fireworks-ai function-calling eval (pass@1): {pass_rate}")
    elif args.dpo == "True":
        print(f"train sample function-calling eval (pass@1): {pass_rate}")

    #results_path = '/home/interstellarninja/ai_projects/axolotl/examples/phi/eval_results.json'
    results_path = './eval_results.json'
    with open(results_path, 'w') as file:
        json.dump(model_evaluator.eval_results, file)

    if args.dpo == "True":
        dpo_path = '/dpo_selfgen.json'
        with open(dpo_path, 'w') as file:
            json.dump(model_evaluator.dpo_results, file)
