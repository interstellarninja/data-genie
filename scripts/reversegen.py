import os
import re
import utils
import time
import yaml
import csv
import json
import argparse
import datetime
import threading
import concurrent.futures
from itertools import islice
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain.schema import Document

from aiutilities import AIUtilities
from schema import OutputSchema
from promptmanager import PromptManager
from search import WebSearch
from vectordb import VectorDB

from dotenv import load_dotenv
load_dotenv()

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and set the logging level
file_handler = logging.FileHandler('reverse_generator.log')
file_handler.setLevel(logging.DEBUG)  # Set the desired logging level for the file handler

# Create a console handler and set the logging level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the desired logging level for the console handler

# Create a formatter and attach it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class DataGenPipeline:
    def __init__(self, config_path):
        # load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.ai_utilities = AIUtilities()
        self.web_search_client = WebSearch()
        self.vector_db = None
        #self.promptmanager = PromptManager(self.config)
        self.file_write_lock = threading.Lock()

    def retrieve_and_combine_documents(self, query, num_results, folder_path, char_limit):
        # Check if the folder already exists
        if os.path.exists(folder_path) and os.listdir(folder_path):
            # Read from existing JSON files
            search_results = utils.read_documents_from_folder(folder_path, num_results)
        else:
            # Fetch new search results
            # Retrieve Google search results
            google_results = self.web_search_client.google_search(query, num_results)
            # Combine results to avoid duplicate URLs
            combined_results = [url for url in google_results]
            
            try:
                bing_results = self.web_search_client.bing_web_search(query, num_results)
                # Add Bing results without duplicate URLs to the combined results
                for url in bing_results:
                    if url not in combined_results:
                        combined_results.append(url)
            except Exception as e:
                logger.info(f"Could not complete bing search: {e}")
           
            # Fetch and save new search results
            search_results = self.web_search_client._scrape_results_parallel(combined_results)
            utils.save_search_results(folder_path, search_results)
            logger.info(f"Search results saved successfully at {folder_path}")

        try:
            combined_text = utils.combine_search_result_documents(search_results, char_limit)
            return combined_text
        except Exception as e:
            return f"Exception in the loop: {e}"
    
    def retrieve_and_combine_examples(self, query, results_path, num_examples=2):
        # Create an instance of the VectorDB class
        if not self.vector_db:
            self.vector_db = VectorDB()
            schema_path = self.config["paths"]["redis_schema"]
            examples_path = self.config["paths"]["examples_path"]

            # Try loading the existing VectorDB
            try:
                if not self.vector_db.load_vector_store(schema_path):
                    # Loading failed, so initialize VectorDB
                    print("Loading existing VectorDB failed. Initializing...")
                    self.vector_db.initialize_vector_store(examples_path, schema_path)
                    if os.path.exists(results_path) and os.listdir(results_path):
                        documents = self.vector_db.load_documents_from_folder(results_path)
                        self.vector_db.rds.add_documents(documents)
                else:
                    print("Existing VectorDB loaded successfully.")
            except Exception as e:
                print("Exception occurred:", e)
                logger.info(f"Couldn't load existing index: {e}")

        retrieved_docs = self.vector_db.perform_similarity_search(query, num_examples)
        combined_examples = utils.combine_examples(retrieved_docs, type="reversegen")
        return combined_examples

    def save_and_index_results(self, file_path, completion, task_desc):
        try:
            with self.file_write_lock:
                with open(file_path, 'w') as json_file:
                    json.dump(completion, json_file, indent=2)
                logger.debug(f"Lock released for {task_desc}")

            logger.info(f"Results for {task_desc} saved successfully at {file_path}")

            # index the result to vectordb for example selection
            document = Document(
                page_content=completion,
                metadata={
                    "source": file_path
                }
            )
            self.vector_db.rds.add_documents([document])

        except Exception as e:
            logger.debug(f"Error extracting and saving results for {task_desc}: {str(e)}")
        finally:
            # Ensure that the lock is always released
            self.file_write_lock.release()

    def run_generation_prompt(self, variables, tools, prompt_type, json_object=False):
        prompt_yaml_path = self.config["paths"][prompt_type]
        prompt_manager = PromptManager(self.config)
        prompt_text = prompt_manager.generate_prompt(variables, prompt_yaml_path)
        messages = [
            {"role": "user", "content": prompt_text}
        ]  
        response = self.ai_utilities.run_ai_tool_completion(messages, tools, tool_choice="none", json=json_object) 
        return response.content
    
    @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(3))
    def run_data_generation(self, task, query, ai_vendor, num_results):
        # search results folder path
        today_date = datetime.date.today()
        folder_name = f"search_results/{today_date}"
        folder_path = os.path.join(os.getcwd(), folder_name, task[0], task[1], task[2])
        folder_path = folder_path.replace(' ', '_')
        os.makedirs(folder_path, exist_ok=True)
        
         # Create a folder for each task if it doesn't exist
        task_desc = f"{task[0]}_{task[1]}_{task[2]}"
        task_desc = task_desc.replace(' ', '_')
        results_path = self.config["paths"]["results_generated"]
        #results_path = f"{results_path}/{ai_vendor}_{today_date}"
        corrected_results_path = os.path.join(os.getcwd(), f"{results_path}_corrected", task[0], task[1])
        results_path = os.path.join(os.getcwd(), results_path, task[0], task[1])
        results_path = results_path.replace(' ', '_')
        corrected_results_path = corrected_results_path.replace(' ', '_')
        os.makedirs(results_path, exist_ok=True)

        file_path = os.path.join(results_path, f"{task[2]}.json")
        file_path = file_path.replace(' ', '_')
        corrected_file_path = os.path.join(corrected_results_path, f"{task[2]}.json")
        corrected_file_path = corrected_file_path.replace(' ', '_')
        # Check if the file already exists
        if not os.path.exists(corrected_file_path):
            ctx_len = self.ai_utilities.get_ai_context_length(ai_vendor)
            char_limit = (int(ctx_len) - 8000) * 4
            logger.info(f"The character limit for documents is:{char_limit}")
            combined_documents = self.retrieve_and_combine_documents(query, num_results, folder_path, char_limit)
            combined_examples = self.retrieve_and_combine_examples(query, corrected_results_path, num_examples=2)
            # Set variables for prompt YAML
            variables = {
                "category": task[0],
                "subcategory": task[1],
                "task": task[2],
                "doc_list": combined_documents,
                "examples": combined_examples
                #"pydantic_schema": OutputSchema.schema_json(),
            }

            with open(file_path) as f:
                json_file = json.load(f)

            messages = json_file["messages"]
            tool_messages = []
            for message in messages:
                if message["role"] == "user":
                    user_message = message["content"]
                elif message["role"] == "assistant":
                    assistant_message = message["content"]
                elif message["role"] == "tool":
                    tool_messages.append(message)
            
            tools = []
            for tool in json_file["tools"]: 
                tools.append(utils.fix_tools_format(tool))

            user_query_vars = {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "tool_call_results": tool_messages 
            }
            user_query_vars = {**variables, **user_query_vars}
            # generate user query given function calls
            user_query = self.run_generation_prompt(user_query_vars, tools, prompt_type="prompt_user_query")
            print(user_query)
            # run function call completion
            func_call_messages = [
                {"role": "user", "content": f"{user_query}"}
            ]
            tool_call_response = self.ai_utilities.run_ai_tool_completion(func_call_messages, tools, tool_choice="auto")
            print(tool_call_response)
            tool_call_message = {key: value for key,  value in tool_call_response.model_dump().items() if value is not None}
            func_call_messages.append(tool_call_message)

            func_result_vars = variables.update({
                "user_message": user_query,
                "assistant_message": tool_call_response.tool_calls,
                "tool_call_results": tool_messages 
            })

            # generate function results
            function_results = self.run_generation_prompt(func_result_vars, tools, prompt_type="prompt_func_results", json_object=True)
            function_results = json.loads(function_results)
            print(function_results)
            for tool in function_results['tools']:
                tool['content'] = json.dumps(tool['content'])
                func_call_messages.append(tool)

            summary_response = self.ai_utilities.run_ai_tool_completion(func_call_messages, tools, tool_choice="none")
            print(summary_response)
            func_call_messages.append(json.loads(summary_response.model_dump_json()))
            logger.info(f"Here's the generated json output:\n{func_call_messages}")
            # Extract and save results for each task
            logger.info(f"saving json files for {task_desc}")
            conversations = {"messages":func_call_messages, "tools":tools}
            self.save_and_index_results(corrected_file_path, conversations, task_desc)

            return conversations
        else:
            return f"Data already generated for the {task_desc}"
        
    def run_generation_pipeline(self, ai_vendor="openai", num_results=10, num_tasks=5):
        curriculum_csv_path = self.config["paths"]["curriculum_csv"]
        with open(curriculum_csv_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            tasks = [(row['Category'], row['SubCategory'], row['Task']) for row in islice(reader, num_tasks)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {executor.submit(self.run_data_generation, task, utils.generate_query(*task), ai_vendor, num_results): task for task in tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    completion = future.result()
                    logger.info(f"Category: {task[0]}, SubCategory: {task[1]}, Task: {task[2]}")
                    logger.info("Completion: {}".format(completion))
                except Exception as e:
                    logger.error(f"Error processing task {task[0]}: {str(e)}")
                # Introduce a small delay between tasks (e.g., 0.1 seconds)
                time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data generation pipeline")
    parser.add_argument("--ai_vendor", choices=["openai", "anthropic", "together", "anyscale"], default="openai", help="choose AI vendor (openai, anthropic, together, anyscale)")
    parser.add_argument("--num_results", type=int, default=10, help="Number of top-k documents for search results")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to generate data for")

    args = parser.parse_args()

  # Example usage for running analysis for companies in a CSV file
    config_path = "/Users/air/Documents/agi_projects/function_calling/config.yaml"
    datagen = DataGenPipeline(config_path)
    datagen.run_generation_pipeline(ai_vendor=args.ai_vendor, num_results=args.num_results, num_tasks=args.num_tasks)
