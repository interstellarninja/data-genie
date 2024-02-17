# utility functions
import ast
import json
import os
import re
from pydantic import ValidationError
import yaml

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_yaml(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def generate_query(task):
    return f'"{task["Category"]}" "{task["SubCategory"]}" {task["Task"]} AND (documentation OR functions OR examples OR code)'

def combine_search_result_documents(search_results, char_limit):
    combined_text = ''
    character_count = 0

    for item in search_results:
        if item is not None and "url" in item and "content" in item:
            url = item.get("url", "")
            content = item.get("content", "")

            # Remove special characters from content
            cleaned_content = remove_special_characters(content)

            if "tables" in item:
                cleaned_content += f"\n{convert_tables_to_markdown(item['tables'])}"

            # Check if appending the current document exceeds the token limit
            if character_count + len(cleaned_content) > char_limit:
                print(f"Character limit reached. Stopping further document append.")
                break

            # Update the character count with the characters from the current document
            character_count += len(cleaned_content)

            combined_text += f'<doc index="{url}">\n'
            combined_text += f'{cleaned_content}'

            combined_text += '\n</doc>\n'
            logger.info(f"Document from {url} added to the combined text")
    return combined_text

def convert_tables_to_markdown(tables):
    markdown = ""
    for table in tables:
        markdown += "|"
        for header in table[0]:
            markdown += f" {header} |"
        markdown += "\n|"
        for _ in table[0]:
            markdown += " --- |"
        markdown += "\n"
        for row in table[1:]:
            markdown += "|"
            for cell in row:
                markdown += f" {cell} |"
            markdown += "\n"
    return markdown

def combine_examples(docs, type=None):
    examples = ""
    for doc in docs:
        examples += f"<example = {os.path.basename(doc.metadata['source'])}>\n"
        if type == "reversegen":
            examples += f"{json.loads(doc.page_content)['messages']}"
        else:
            examples += f"{doc.page_content}"
        examples += f"\n</example>\n"
    return examples

def read_documents_from_folder(folder_path, num_results):
    search_results = []

    if os.path.exists(folder_path) and os.listdir(folder_path):
        # Read from existing JSON files
        for i, filename in enumerate(os.listdir(folder_path)):
            if i < num_results and filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    result_data = json.load(file)
                    search_results.append(result_data)
        return search_results
    else:
        return "No files in the directory"

def save_search_results(folder_path, search_results):
    os.makedirs(folder_path, exist_ok=True)

    for i, item in enumerate(search_results):
        if item is not None and "url" in item and "content" in item:
            file_path = os.path.join(folder_path, f"result_{i}.json")
            result_data = {"url": item["url"], "content": item["content"], "tables": item["tables"]}

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(result_data, file, ensure_ascii=False, indent=2)
            
            logger.info(f"Websearch results from {item['url']} saved")

def remove_special_characters(input_string):
    # Use a regular expression to remove non-alphanumeric characters (excluding spaces)
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    return cleaned_string

def extract_json_from_response(response_string):
    try:
        # Load the JSON data
        start_index = response_string.find('{')
        end_index = response_string.rfind('}') + 1
        json_data = response_string[start_index:end_index]
        json_data = json.loads(json_data)

        # Parse the JSON data using the OutputSchema
        #output_schema = OutputSchema.model_validate(json_data)

        #return output_schema.model_dump()
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except ValidationError as e:
        print(f"Error validating JSON against schema: {e}")
        return None

def convert_enum_to_list(prop_data):
    if "enum" in prop_data:
        enum_value = prop_data["enum"]
        if isinstance(enum_value, dict):
            prop_data["enum"] = list(enum_value.keys())
        elif not isinstance(enum_value, list):
            prop_data["enum"] = [enum_value]  # Convert to a list
            
def fix_tools_format(tool):

    if "type" not in tool or tool["type"] != "function":
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        for prop_name, prop_data in properties.items():
            convert_enum_to_list(prop_data)

        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    else:
        parameters = tool.get("function", {}).get("parameters", {})
        properties = parameters.get("properties", {})
        for prop_name, prop_data in properties.items():
            convert_enum_to_list(prop_data)

        return tool

def get_assistant_message(completion):
    message  = json.loads(completion)['choices'][0]['message']
    if message['tool_calls']:
        tool_calls = []
        for tool_call in message['tool_calls']:
            tool_calls.append(tool_call['function'])
        return tool_calls
    else:
        return message['content']
    
def extract_toolcall_code_blocks(content):
    # Define the pattern to find all tool_call blocks
    pattern = r"```tool_call\s*({.*?})\s*```"

    # Find all matches
    matches = re.findall(pattern, content, re.DOTALL)

    # Process the matches
    result = []
    for match in matches:
        try:
            # Load as JSON
            json_data = ast.literal_eval(match)
            result.append(json_data)
        except Exception as e:
            print(f"Error processing block {match}: {e}")
    return result

def extract_tool_code_block(content):
     # Define the pattern to find all tool_call blocks
    pattern = r"```tools\s*({.*?})\s*```"

    # Find all matches
    match = re.search(pattern, content, re.DOTALL)

    # Process the matches
    result = None
    if match:
        try:
            # Load as JSON
            json_data = ast.literal_eval(match.group(1))
            result = json_data
        except Exception as e:
            print(f"Error processing block {match.group(0)}: {e}")
    return result

def strip_incomplete_text(text):
    # Find the last occurrence of a full stop
    last_full_stop_index = text.rfind('.')
    
    if last_full_stop_index != -1:  # If a full stop is found
        # Return the text up to the last full stop
        return text[:last_full_stop_index+1]  # Include the full stop
    else:
        # If no full stop is found, return the original text
        return text

def clean_file_path(file_path):
    # Remove special characters
    cleaned_file_path = re.sub(r'[^\w\s-]', '', file_path)
    
    # Replace spaces with underscores
    cleaned_file_path = cleaned_file_path.replace(' ', '_')
    
    # Shorten the file path if it exceeds 255 characters
    if len(cleaned_file_path) > 255:
        base_path, file_name = os.path.split(cleaned_file_path)
        file_name = file_name[:255 - len(base_path) - 1]  # Subtract 1 for the separator
        cleaned_file_path = os.path.join(base_path, file_name)
    
    return cleaned_file_path


