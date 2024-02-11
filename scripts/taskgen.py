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
            folder_code = {}

            for item in folder_contents:
                # Check if the item is a file and if its extension is .py, .ipynb, or .md
                if item['type'] == 'file' and item['name'].endswith(('.py', '.ipynb', '.md')):
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
                        parent_folder = os.path.basename(os.path.dirname(item['path']))
                        if parent_folder not in folder_code:
                            folder_code[parent_folder] = []
                        # Use XML delimiters with filenames
                        folder_code[parent_folder].append(f"<file={item['name']}>\n{file_content}\n</file>")
                    else:
                        print("Failed to fetch file content:", file_url)
                elif item['type'] == 'dir':
                    subdir_url = item['url']
                    self.extract_code_from_folder(subdir_url, os.path.join(output_folder))

            for folder, code_lines in folder_code.items():
                output_file = os.path.join(output_folder, f"{folder}_code.txt")
                with open(output_file, 'w') as f:
                    f.write('\n\n'.join(code_lines))
        else:
            print("Failed to fetch folder contents. Status code:", response.status_code)


    def process_code_and_generate_tasks(self, extracted_code_folder):
        prompt = "Please extract tasks from given code file in csv format. "
        prompt += "The csv columns are category, subcategory, task. "
        prompt += "The task should be about generating json object representation for components of agentic frameworks. "
        prompt += "The code would usually have task description with key value pairs that will be input or output. "
        prompt += f"You may use the examples prvided below:\n"

        examples = [
            {
                "category": "Instructor Agents",
                "subcategory": "Data Extraction",
                "task": "Generate a json representation for a data extraction system that processes markdown tables from text and images. The system should convert markdown data into pandas DataFrames, apply necessary pre-processing such as stripping whitespaces, and ensure the output is tidy without merging separate tables. Additionally, the system should be capable of interpreting images containing data tables, extracting the information, and representing it in markdown format with appropriate headers and descriptions."
            },
            {
                "category": "Instructor Agents",
                "subcategory": "Task Management",
                "task": "Generate a json representation for a task management system that updates an existing task to create 20 new GIFs for the Taskbot page on a personal site and creates a new task to create an additional 20 animated GIFs after the initial task is completed. The system should handle task dependencies and include notes on the user's strategy for task completion."
            },
            {
                "category": "Instructor Agents",
                "subcategory": "Summarization",
                "task": "Generate a json representation for a summarization process that involves creating increasingly concise and entity-dense summaries of an article. The process should include initial verbose summaries and subsequent rewrites that integrate missing entities while maintaining the same length, ensuring a minimum entity density and no omission of previously mentioned entities."
            },
            {
                "category": "Instructor Agents",
                "subcategory": "Knowledge Graph Generation",
                "task": "Generate a json object representation for a Knowledge Graph generation system that iteratively builds a graph from text chunks. The system should deduplicate nodes and edges, reuse nodes when possible, and provide visual output of the graph after each iteration and upon completion."
            },
            {
                "category": "Instructor Agents",
                "subcategory": "Semantic Search",
                "task": "Generate a json object for a semantic search system that segments search requests into a set of detailed and comprehensive queries. The system should provide tips for expanding queries, use titles to explain expected results, and ensure queries are specific and broad for effective search."
            }
        ]

        for i, example in enumerate(examples):
            prompt += f"<example={i}> {example} </example>\n"
        
        csv_list = []
        json_path = os.path.join(extracted_code_folder, "task_json")
        os.makedirs(json_path, exist_ok=True)

        # Function to process a single file
        def process_file(filename):
            csv_list = []
            file_path = os.path.join(extracted_code_folder, filename)
            with open(file_path, 'r') as file:
                code = file.read()
                code = code[:250000]
            
            # Check if the JSON file already exists
            json_file_path = os.path.join(json_path, f"{filename[:-3]}.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as jsonfile:
                    task = json.load(jsonfile)
            else:
                code_prompt = f"Please use the following code as reference for the task: {code}\n"
                schema_prompt = f"Return a single json object that adheres to the following pydantic json schema properties for your output:\n{Task.schema_json()}\n"
                schema_prompt += "Do not return the pydantic schema as it is but json object with category, subcategory, and task keys"
                final_prompt = prompt + code_prompt + schema_prompt
                print(final_prompt)
                task = self.ai_utils.run_openai_completion(final_prompt)
                task = json.loads(task)
                print(task)
                # Write the task JSON to file for future use
                with open(json_file_path, 'w') as jsonfile:
                    json.dump(task, jsonfile, indent=4)

            csv_list.append(task)
            query = generate_query(task['category'], task['subcategory'], task['task'])
            task_list = list(task.values())
            print(f"Generating for task subcategory: {task_list[1]}")
            completion = self.datagenerator.run_data_generation(task_list, query, ai_vendor="openai", num_results=10, combined_documents=code)
            print(completion)

        # Use ThreadPoolExecutor to parallelize file processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_file, os.listdir(extracted_code_folder))

        self.convert_to_csv(csv_list, os.path.join(extracted_code_folder, 'tasks.csv'))

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
    args = parser.parse_args()

    config_path = "./config.yaml"
    task_generator = TaskGenerator(config=config_path)
    if not os.path.exists(args.documents_folder):
        os.makedirs(args.documents_folder)
    task_generator.extract_code_from_folder(args.repo_url, args.documents_folder)
    task_generator.process_code_and_generate_tasks(args.documents_folder)

class Task(BaseModel):
    category: str
    subcategory: str
    task: str

# Define examples and prompt here

if __name__ == "__main__":
    main()
