from pydantic import BaseModel
from typing import List, Dict
from schema import OutputSchema
import yaml
import json
import csv
import os

class PromptSchema(BaseModel):
    Role: str
    Objective: str
    Guidelines: str
    Documents: str
    Examples: str
    Output_instructions: str 
    Output_schema: str
    Assistant: str

class PromptManager:
    def __init__(self, config):
        self.config = config
        
    def format_yaml_prompt(self, prompt_schema: PromptSchema, variables: Dict) -> str:
        formatted_prompt = ""
        for field, value in prompt_schema.dict().items():
            formatted_value = value.format(**variables)
            formatted_prompt += f"{field}:\n{formatted_value}\n"
        return formatted_prompt

    def read_yaml_file(self, file_path: str) -> PromptSchema:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        
        prompt_schema = PromptSchema(
            Role=yaml_content.get('Role', ''),
            Objective=yaml_content.get('Objective', ''),
            Guidelines=yaml_content.get('Guidelines', ''),
            Documents=yaml_content.get('Documents', ''),
            Examples=yaml_content.get('Examples', ''),
            Output_instructions=yaml_content.get('Output_instructions', ''),
            Output_schema=yaml_content.get('Output_schema', ''),
            Assistant=yaml_content.get('Assistant', '')
        )
        return prompt_schema
    
    def generate_prompt(self, variables, prompt_yaml_path):
        prompt_schema = self.read_yaml_file(prompt_yaml_path)
        prompt = self.format_yaml_prompt(prompt_schema, variables)
        return prompt

    def generate_system_prompt(self, prompt_schema: PromptSchema, output_schema: str) -> str:
        system_content = f"{prompt_schema.Role}\n{prompt_schema.Guidelines}\n{prompt_schema.Output_instructions}\n{output_schema}"
        return system_content

    def generate_user_prompt(self, prompt_schema: PromptSchema, variables: Dict) -> str:
        user_content = prompt_schema.Objective.format(**variables) + "\n" + \
                       prompt_schema.Documents.format(**variables) + "\n" + \
                       prompt_schema.Examples.format(**variables) + "\n" + \
                       prompt_schema.Assistant.format(**variables)
        return user_content

    def generate_prompt_messages(self, variables, prompt_yaml_path, system_prefix=None):
        prompt_schema = self.read_yaml_file(prompt_yaml_path)
        output_schema = variables.get("pydantic_schema", "")
        if system_prefix:
            system_prompt = system_prefix + self.generate_system_prompt(prompt_schema, output_schema)
        else:
            system_prompt = self.generate_system_prompt(prompt_schema, output_schema)
        user_prompt = self.generate_user_prompt(prompt_schema, variables)
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

# Example usage:
# Example usage:
if __name__ == "__main__":
    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    prompt_manager = PromptManager(config)

    generation_type = "function_calling"
    
    examples_json_path = os.path.join(prompt_manager.config["paths"]["examples_path"], generation_type)
    examples = ""
    for file in os.listdir(examples_json_path):
        file_path = os.path.join(examples_json_path, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                examples += f"<example = {file}>\n"
                examples += f"{json.load(f)}"
                examples += f"\n</example>\n"
    
    curriculum_csv_path = os.path.join(prompt_manager.config["paths"]["curriculum_csv"], f"{generation_type}.csv")
    with open(curriculum_csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        tasks = [(row['Category'], row['SubCategory'], row['Task']) for row in reader]
        task = tasks[12]
    
    query = f"{task[0]}, {task[1]}, {task[2]}, functions, APIs, documentation"
    variables = {
        "category": task[0],
        "subcategory": task[1],
        "task": task[2],
        "doc_list": "list of documents",
        "examples": examples,
        "pydantic_schema": OutputSchema.schema_json(),
    }

    prompt_messages = prompt_manager.generate_prompt_messages(variables, "./prompt_assets/prompts/function_calling.yaml")
    print(json.dumps(prompt_messages, indent=4))
