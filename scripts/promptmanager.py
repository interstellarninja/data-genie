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
        #print(file_path)
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
    
    def generate_prompt(self, variables):
        prompt_yaml_path = self.config["paths"]["prompt_yaml"]
        prompt_schema = self.read_yaml_file(prompt_yaml_path)
        print(prompt_schema)
        prompt = self.format_yaml_prompt(prompt_schema, variables)

        return prompt

# Example usage:
if __name__ == "__main__":
    # Create an instance of PromptManager
    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    prompt_manager = PromptManager(config)
    
    examples_json_path = prompt_manager.config["paths"]["examples_path"]
    examples = ""
    for file in os.listdir(examples_json_path):
        file_path = os.path.join(examples_json_path, file)
        with open(file_path, "r") as f:
            examples += f"<example = {file}>\n"
            examples += f"{json.load(f)}"
            examples += f"\n</example>\n"
    
    curriculum_csv_path = prompt_manager.config["paths"]["curriculum_csv"]
    with open(curriculum_csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        tasks = [(row['Category'], row['SubCategory'], row['Task']) for row in reader]
        task = tasks[12]
    
    query = f"{task[0]}, {task[1]}, {task[2]}, functions, APIs, documentation"
    # Set variables for YAML
    variables = {
        "category": task[0],
        "subcategory": task[1],
        "task": task[2],
        "doc_list": "list of documents",
        "examples": examples,
        "pydantic_schema": OutputSchema.schema_json(),
    }

    # Format the YAML prompt
    #formatted_prompt = prompt_manager.format_yaml_prompt(prompt_schema, variables)
    formatted_prompt = prompt_manager.generate_prompt(variables)

    # Print the formatted prompt
    print(formatted_prompt)