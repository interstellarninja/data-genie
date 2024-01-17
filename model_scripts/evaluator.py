import torch
import json
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

class ModelEvaluator:
    def __init__(self, model_path):
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

    def validate_and_extract_tool_call(self, completion):
        pattern = re.compile(r'<\|assistant\|>([\s\S]*)')

        match = pattern.search(completion)

        if match:
            content_after_assistant = match.group(1)
            if content_after_assistant is not None:
                content = content_after_assistant.strip()
                # Remove unwanted characters
                content = re.sub(r'\\1', '', content)
                content = re.sub(r'\\n', '', content)
                # Extract code block
                start = content.find("```tool_call")
                if start >= 0:
                    end = content.find("```", start + 1)
                    code_block = content[start + 12:end]
                    try:
                        # Replace single quotes with double quotes
                        code_block = code_block.replace("'", "\"")
                        # Load as JSON directly
                        json_data = json.loads(code_block)
                        return True, json_data
                    except Exception as e:
                        print(f"Here's the extracted block {code_block} with error {e}")
                        return False, content
                else:
                    print("Did not find tool_call")
                    return False, content
        else:
            print("No match found for the pattern.")

    def validate_func_calls(self, generated_values, expected_values):
        for key, expected_value in expected_values.items():
            # Handle nested key
            keys = key.split(".")
            obj = generated_values

            try:
                for key in keys[:-1]:
                    obj = obj.get(key, {})
            except AttributeError:
                print(f"AttributeError: 'NoneType' object has no attribute 'get'")
                return "failed"

            value = obj.get(keys[-1])

            # Compare value
            if value != expected_value:
                print(f"function call argument values do not match")
                return "failed"

        return "passed"

    def evaluate_dataset(self, eval_dataset):

        for sample in eval_dataset:
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

            validation, assistant_message = self.validate_and_extract_tool_call(completion)
            print(assistant_message)

            if validation:
                result = self.validate_func_calls(assistant_message, json.loads(sample['completion']))
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
    # Set your model path
    model_path = '/home/interstellarninja/ai_projects/axolotl/examples/stablelm/interstellarninja/stablelm-zephyr-3b-func-calling-dpo'
    
    # Load evaluation dataset
    eval_dataset = load_dataset("NousResearch/func-calling-eval")['train']

    # Create model evaluator instance
    model_evaluator = ModelEvaluator(model_path)

    # Evaluate the dataset
    model_evaluator.evaluate_dataset(eval_dataset)
    results_path = '/home/interstellarninja/ai_projects/axolotl/examples/stablelm/eval_results.json'
    with open(results_path, 'w') as file:
        json.dump(model_evaluator.eval_results, file)

    # Calculate and print pass rate
    pass_rate = model_evaluator.calculate_pass_rate()
    print(f"fireworks-ai function-calling eval (pass@1): {pass_rate}")
