from pydantic import BaseModel
from typing import Dict
import json

from utils import (
    load_yaml,
    get_fewshot_examples
)

class SysPromptSchema(BaseModel):
    Role: str
    Objective: str
    Tools: str
    Examples: str
    Schema: str
    Instructions: str 

class FunctionCall(BaseModel):
    arguments: dict
    name: str

class SysPromptManager:
    def __init__(self, config_path):
        self.config = load_yaml(config_path)
        
    def format_yaml_prompt(self, prompt_schema: SysPromptSchema, variables: Dict) -> str:
        formatted_prompt = ""
        for field, value in prompt_schema.dict().items():
            if field == "Examples" and variables.get("examples") is None:
                continue
        
            formatted_value = value.format(**variables)
            if field == "Instructions":
                formatted_prompt += f"{formatted_value}\n"
            else:
                formatted_value = formatted_value.replace("\n", " ")
                formatted_prompt += f"{formatted_value}"
            
        return formatted_prompt

    def read_prompt_yaml_file(self, file_path: str) -> SysPromptSchema:
        yaml_content = load_yaml(file_path)
        prompt_schema = SysPromptSchema(
            Role=yaml_content.get('Role', ''),
            Objective=yaml_content.get('Objective', ''),
            Tools=yaml_content.get('Tools', ''),
            Examples=yaml_content.get('Examples', ''),
            Schema=yaml_content.get('Schema', ''),
            Instructions=yaml_content.get('Instructions', ''),
        )
        return prompt_schema
    
    def generate_sys_prompt(self, sample, num_fewshot=None):
        prompt_path = self.config['paths']['sys_prompt_yaml']
        prompt_schema = self.read_prompt_yaml_file(prompt_path)

        example_path = self.config['paths']['fewshot_path']
        if num_fewshot is not None:
            examples = get_fewshot_examples(num_fewshot, example_path)
        else:
            examples = None

        schema_json = json.loads(FunctionCall.schema_json())

        variables = {
            "tools":sample["tools"],
            "examples": examples,
            "schema": schema_json
        }
        sys_prompt = self.format_yaml_prompt(prompt_schema, variables)
        return sys_prompt
        
        
