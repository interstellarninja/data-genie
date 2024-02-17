import csv
import json
import os
import requests
import argparse
import concurrent.futures
from pydantic import BaseModel
from utils import generate_query
from aiutilities import AIUtilities
from datagen import DataGenPipeline
from validator import validate_json_data, validate_json_object
from tenacity import retry, stop_after_attempt, wait_fixed

class TaskGenerator: 
    def __init__(self, config):
        self.session = requests.Session()
        self.ai_utils = AIUtilities()
        self.datagenerator = DataGenPipeline(config, type="json_mode")
        self.datagenerator.initialize_vector_db()
    
    def extract_code_from_folder(self, folder_url, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        response = self.session.get(folder_url)
        if response.status_code == 200:
            folder_contents = response.json()

            for item in folder_contents:
                # Check if the item is a file and if its extension is .py, .ipynb, or .txt
                if item['type'] == 'file' and item['name'].endswith(('.py', '.ipynb', '.txt', '.js')):
                    # Skip if the file name is __init__.py
                    if item['name'] == '__init__.py':
                        print(f"Skipping __init__.py file: {item['name']}")
                        continue

                    output_file = os.path.join(output_folder, item['name'])
                    if os.path.exists(output_file):
                        print(f"File already exists: {output_file}. Skipping.")
                        continue
                    file_url = item['download_url']
                    file_content_response = self.session.get(file_url)
                    if file_content_response.status_code == 200:
                        file_content = file_content_response.text
                        # Skip if the file content is empty or only contains whitespace
                        if not file_content.strip():
                            print(f"Skipping empty file: {item['name']}")
                            continue
                        # Save each file directly into the output folder
                        with open(output_file, 'w') as f:
                            f.write(file_content)
                    else:
                        print("Failed to fetch file content:", file_url)
                elif item['type'] == 'dir':
                    subdir_url = item['url']
                    self.extract_code_from_folder(subdir_url, output_folder)  # Recursive call with the same output_folder

        else:
            print("Failed to fetch folder contents. Status code:", response.status_code)

    #def extract_code_from_folder(self, folder_url, output_folder):
    #    os.makedirs(output_folder, exist_ok=True)
    #    response = self.session.get(folder_url)
    #    if response.status_code == 200:
    #        folder_contents = response.json()
    #        folder_code = {}
#
    #        for item in folder_contents:
    #            # Check if the item is a file and if its extension is .py, .ipynb, or .md
    #            if item['type'] == 'file' and item['name'].endswith(('.py', '.ipynb', '.txt')):
    #                # Skip if the file name is __init__.py
    #                if item['name'] == '__init__.py':
    #                    print(f"Skipping __init__.py file: {item['name']}")
    #                    continue
#
    #                output_file = os.path.join(output_folder, item['name'])
    #                if os.path.exists(output_file):
    #                    print(f"File already exists: {output_file}. Skipping.")
    #                    continue
    #                file_url = item['download_url']
    #                file_content_response = self.session.get(file_url)
    #                if file_content_response.status_code == 200:
    #                    file_content = file_content_response.text
    #                    # Skip if the file content is empty or only contains whitespace
    #                    if not file_content.strip():
    #                        print(f"Skipping empty file: {item['name']}")
    #                        continue
    #                    parent_folder = os.path.basename(os.path.dirname(item['path']))
    #                    if parent_folder not in folder_code:
    #                        folder_code[parent_folder] = []
    #                    # Use XML delimiters with filenames
    #                    folder_code[parent_folder].append(f"<file={item['name']}>\n{file_content}\n</file>")
    #                else:
    #                    print("Failed to fetch file content:", file_url)
    #            elif item['type'] == 'dir':
    #                subdir_url = item['url']
    #                self.extract_code_from_folder(subdir_url, os.path.join(output_folder))
#
    #        for folder, code_lines in folder_code.items():
    #            output_file = os.path.join(output_folder, f"{folder}_code.txt")
    #            with open(output_file, 'w') as f:
    #                f.write('\n\n'.join(code_lines))
    #    else:
    #        print("Failed to fetch folder contents. Status code:", response.status_code)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def run_data_generation_with_retry(self, task_list, query, code):
        print("This is a TEST")
        completion = self.datagenerator.run_data_generation(task_list, query, ai_vendor="openai", num_results=10, combined_documents=code)
        for message in completion["messages"]:
            role = message["role"]
            content = message.get("content", "")         
            if role == "assistant":
                json_object = content
        schema_dict = completion["pydantic_schema"]
        try:
            is_valid = validate_json_data(json_object, schema_dict)
            if is_valid:
                return json_object  # Return the valid JSON object
            else:
                raise ValueError("Invalid JSON object received from data generation")
        except Exception as e:
            print(f"{e}")
            print(completion)
            raise  # Re-raise the exception to trigger retry

    def generate_task_prompt(self):
        prompt = "Please extract tasks from given code file as a json object. "
        prompt += "The keys of the json object should be category, subcategory, task and schema."
        prompt += "The task should be about generating a json object representation to assist with a user query using agentic frameworks."
        prompt += "The code would usually have task description, pydantic model, json schema and json objects with key value pairs that will be input or output."
        prompt += "The task should not just be about creating a schema since schema would be provided but a json object adhering to the schema that assists with user query."
        prompt += f"You may use the examples provided below as reference:\n"

        examples = [
            {"Category":"WebBrowser Agent","SubCategory":"Planning","Task":"Return a JSON object for agent planning over time","Schema":{"type":"object","properties":{"Planning":{"type":"object","properties":{"short_term":{"type":"array","items":{}},"long_term":{"type":"array","items":{}}}},"required":["Planning"]}}},
            {"Category":"WebBrowser Agent","SubCategory":"Coordination","Task":"Return a JSON object for agent coordination","Schema":{"type":"object","properties":{"Coordination":{"type":"object","properties":{"agent1":{"type":"object","additionalProperties":{}},"agent2":{"type":"object","additionalProperties":{}},"shared_goal":{"type":"string"},"coordination_events":{"type":"array","items":{}}}},"required":["Coordination"]}}},
            {"Category":"WebBrowser Agent","SubCategory":"Information Diffusion","Task":"Return a JSON object for information diffusion","Schema":{"type":"object","properties":{"Information Diffusion":{"type":"object","properties":{"original_agent":{"type":"string"},"origin_event":{"type":"string"},"diffused_events":{"type":"array","items":{}}}},"required":["Information Diffusion"]}}},
            {"Category":"WebBrowser Agent","SubCategory":"Reflection Tree","Task":"Return a JSON object representing an agent's tree of reflections with leaf nodes as base observations and non-leaf nodes as higher-level, more abstract thoughts","Schema":{"type":"object","properties":{"Reflection Tree":{"type":"object","properties":{"leaf_nodes":{"type":"array","items":{"type":"string"}},"non_leaf_nodes":{"type":"object","additionalProperties":{"type":"array","items":{"type":"string"}}}}},"required":["Reflection Tree"]}}},
            {"Category":"WebBrowser Agent","SubCategory":"Spatial Memory Subgraph","Task":"Return a JSON object representing the part of the world perceived by the agent with root, child, and leaf nodes","Schema":{"type":"object","properties":{"Spatial Memory Subgraph":{"type":"object","properties":{"root":{"type":"object","additionalProperties":{"type":"object","properties":{"table":{"type":"string"},"chair":{"type":"string"}}}},"houses":{"type":"object","additionalProperties":{"type":"object","properties":{"bedroom":{"type":"object","properties":{"bed":{"type":"string"}}}}}}}},"required":["Spatial Memory Subgraph"]}}},
            {"Category":"WebBrowser Agent","SubCategory":"Sandbox Environment","Task":"Return a JSON object representing the sandbox world as a tree with areas and objects","Schema":{"type":"object","properties":{"Sandbox Environment":{"type":"object","properties":{"areas":{"type":"object","additionalProperties":{"type":"object","properties":{"objects":{"type":"object","additionalProperties":{}}}}}},"houses":{"type":"object","additionalProperties":{"type":"object","properties":{"rooms":{"type":"object","additionalProperties":{"type":"object","properties":{"objects":{"type":"object","additionalProperties":{}}}}}}}}}},"required":["Sandbox Environment"]}}

        #    {"Category":"Gollie Agents","SubCategory":"Data Extraction","Task":"Generate a json object to assist with a user task for a data extraction framework that processes markdown tables from text and images. The system should convert markdown data into pandas DataFrames, apply necessary pre-processing such as stripping whitespaces, and ensure the output is tidy without merging separate tables. Additionally, the system should be capable of interpreting images containing data tables, extracting the information, and representing it in markdown format with appropriate headers and descriptions.","Schema":{"type":"object","properties":{"data_extraction":{"type":"object","properties":{"markdown_tables":{"type":"string"},"images_with_tables":{"type":"string"}},"required":["markdown_tables","images_with_tables"]}}}},
        #    
        #    {"Category":"Gollie Agents","SubCategory":"Task Management","Task":"Implement a json representation to complete a user's request using a task management agent that updates an existing task to create 20 new GIFs for the Taskbot page on a personal site and creates a new task to create an additional 20 animated GIFs after the initial task is completed. The system should handle task dependencies and include notes on the user's strategy for task completion.","Schema":{"type":"object","properties":{"task_management":{"type":"object","properties":{"update_existing_task":{"type":"string"},"create_new_task":{"type":"string"},"task_dependencies_handling":{"type":"string"},"user_strategy_notes":{"type":"string"}},"required":["update_existing_task","create_new_task","task_dependencies_handling","user_strategy_notes"]}}}},
        #    
        #    {"Category":"Gollie Agents","SubCategory":"Knowledge Graph Generation","Task":"Build a json object representation for a user task using Knowledge Graph generation process that iteratively builds a graph from text chunks. The system should deduplicate nodes and edges, reuse nodes when possible, and provide visual output of the graph after each iteration and upon completion.","Schema":{"type":"object","properties":{"knowledge_graph_generation":{"type":"object","properties":{"text_chunks":{"type":"string"},"deduplication":{"type":"string"},"node_and_edge_reuse":{"type":"string"},"visual_output":{"type":"string"}},"required":["text_chunks","deduplication","node_and_edge_reuse","visual_output"]}}}},
        #    
        #    {"Category":"Gollie Agents","SubCategory":"Semantic Search","Task":"Create a json object for a user task using semantic search framework that segments search requests into a set of detailed and comprehensive queries. The system should provide tips for expanding queries, use titles to explain expected results, and ensure queries are specific and broad for effective search.","Schema":{"type":"object","properties":{"semantic_search":{"type":"object","properties":{"search_requests_segmentation":{"type":"string"},"query_expansion_tips":{"type":"string"},"expected_results_explanation":{"type":"string"},"query_specificity_and_breadth":{"type":"string"}},"required":["search_requests_segmentation","query_expansion_tips","expected_results_explanation","query_specificity_and_breadth"]}}}},
        #    
        #    {"Category":"Gollie Agents","SubCategory":"Multi-Function Agents","Task":"Assist with a user task with a json object using a multi-function agent that processes both weather inquiries and Google search queries. The agent should be able to handle requests for weather information in different locations and units, as well as perform Google searches based on user queries. The system should be capable of parallel processing and provide responses in an iterable format.","Schema":{"type":"object","properties":{"multi_function_agent":{"type":"object","properties":{"weather_inquiries":{"type":"string"},"google_search_queries":{"type":"string"},"location_units_handling":{"type":"string"},"parallel_processing_capability":{"type":"string"},"iterable_responses":{"type":"string"}},"required":["weather_inquiries","google_search_queries","location_units_handling","parallel_processing_capability","iterable_responses"]}}}}
        ]
        for i, example in enumerate(examples):
            prompt += f"<example={i}> {example} </example>\n"
        
        return prompt

    def process_code_and_generate_tasks(self, extracted_code_folder):
        json_path = os.path.join(extracted_code_folder, "task_json")
        os.makedirs(json_path, exist_ok=True)

        # Function to process a single file or recursively process a directory
        def process_file(filename_or_path):
            print(f"Processing {filename_or_path}")
            if os.path.isfile(filename_or_path):
                with open(filename_or_path, 'r') as file:
                    code = file.read()[:250000]
                    self.process_code(filename_or_path, code, json_path)

        # Use ThreadPoolExecutor to parallelize file processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for item in os.listdir(extracted_code_folder):
                item_path = os.path.join(extracted_code_folder, item)
                if os.path.isfile(item_path):
                    executor.submit(process_file, item_path)
                    
    def process_code(self, filename, code, json_path):

        json_file_name = self.get_json_filename(filename)
        print(json_file_name)
        json_file_path = os.path.join(json_path, json_file_name)
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as jsonfile:
                task = json.load(jsonfile)
        else:
            prompt = self.generate_task_prompt()
            code_prompt = f"Please use the following code as reference for the task: {code}\n"
            schema_prompt = f"Return a single json object that adheres to the following pydantic json schema properties for your output:\n{Task.schema_json()}\n"
            schema_prompt += "Do not return the pydantic schema as it is but json object with category, subcategory, task and schema keys"
            final_prompt = prompt + code_prompt + schema_prompt
            print(final_prompt)
            task = self.ai_utils.run_openai_completion(final_prompt)
            task = json.loads(task)
            print(task)
            try:
                # Write the task JSON to file for future use
                with open(json_file_path, 'w') as jsonfile:
                    json.dump(task, jsonfile, indent=4)
            except Exception as e:
                print("Error occurred while saving the JSON file:", e)

        query = generate_query(task)
        print(f"Generating for task subcategory: {task['SubCategory']}")
        try:
            json_object = self.run_data_generation_with_retry(task, query, code)
            print("Valid JSON object received:", json_object)
        except ValueError as ve:
            print(ve)

    def get_json_filename(self, filename):
    # Function to return JSON filename based on the file extension
        filename = os.path.basename(filename)
        if filename.endswith('.py'):
            return f"{filename[:-3]}.json"
        elif filename.endswith('.ipynb'):
            return f"{filename[:-6]}.json"
        elif filename.endswith('.txt'):
            return f"{filename[:-4]}.json"
        elif filename.endswith('.js'):
            return f"{filename[:-3]}.json"
        else:
            return f"{filename}.json"
        
    def convert_to_csv(self, csv_list, output_file):
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['category', 'subcategory', 'task']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for task in csv_list:
                writer.writerow(task)

def main():
    parser = argparse.ArgumentParser(description="Extract code from a GitHub repository.")
    parser.add_argument("--repo_url", type=str, help="URL of the GitHub repository")
    parser.add_argument("--documents_folder", type=str, help="Output folder for extracted code")
    parser.add_argument("--task_generated", default="False", type=str, help="folder with task json files")
    args = parser.parse_args()

    config_path = "./config.yaml"
    task_generator = TaskGenerator(config=config_path)
    if not os.path.exists(args.documents_folder):
        os.makedirs(args.documents_folder)
    
    if args.task_generated == "True":
        task_generator.process_code_and_generate_tasks(args.documents_folder)
    elif args.task_generated == "False":
        task_generator.extract_code_from_folder(args.repo_url, args.documents_folder)
        task_generator.process_code_and_generate_tasks(args.documents_folder)
    else:
        print("Please enter true or false")

class Task(BaseModel):
    Category: str
    SubCategory: str
    Task: str
    Schema: dict

# Define examples and prompt here

if __name__ == "__main__":
    main()
