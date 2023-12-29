# utility functions
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

def generate_query(category, subcategory, task):
    return f'"{category}" "{subcategory}" {task} AND (documentation OR functions OR examples OR code)'

def combine_search_result_documents(search_results, char_limit):
    combined_text = ''
    character_count = 0

    for item in search_results:
        if item is not None and "url" in item and "content" in item:
            url = item.get("url", "")
            content = item.get("content", "")

            # Remove special characters from content
            cleaned_content = remove_special_characters(content)

            # Check if appending the current document exceeds the token limit
            if character_count + len(cleaned_content) > char_limit:
                print(f"Character limit reached. Stopping further document append.")
                break

            # Update the character count with the characters from the current document
            character_count += len(cleaned_content)

            combined_text += f'<doc index="{url}">\n'
            combined_text += f'{cleaned_content}'

            combined_text += '\n</doc>\n'

def combine_examples(docs):
    examples = ""
    for doc in docs:
        examples += f"<example = {os.path.basename(doc.metadata['source'])}>\n"
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

def convert_tables_to_markdown(tables):
    markdown = ""
    for table in tables:
        markdown = "|"
        for header in table[0]:
            markdown += f" {header[0]} |"
        markdown += "\n|"
        for _ in table[0]:
            markdown += " --- |"
        markdown += "\n"
        for row in table[1:]:
            markdown += "|"
            for cell in row:
                markdown += f" {cell[1]} |"
            markdown += "\n"
    return markdown