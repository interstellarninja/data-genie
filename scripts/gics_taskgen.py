
import json
import pandas as pd
import argparse
from pydantic import BaseModel, Field, EmailStr
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import List
from tenacity import retry, stop_after_attempt, wait_fixed
from aiutilities import AIUtilities

class Task(BaseModel):
    Category: str
    SubCategory: str
    Task: str
    Schema: dict

class TaskJsonLinesList(BaseModel):
    Tasks: List[Task] = Field(..., min_items=5, max_items=5)

class GICSTaskGenerator:
    def __init__(self, excel_path, output_file_path):
        self.df = pd.read_excel(excel_path, skiprows=3)
        self.df = self.df.iloc[:, [1, 3, 5, 7]]
        self.df.columns = ["Sector", "Industry Group", "Industry", "Sub-Industry"]
        self.df = self.df.ffill()
        self.output_file_path = output_file_path

    def row_to_json(self, row):
        return {
            "Category": row["Industry Group"],
            "SubCategory": row['Industry'],
            "Task": row["Sub-Industry"]
        }

    def generate_json_lines(self):
        json_lines = self.df.apply(self.row_to_json, axis=1).to_json(orient='records', lines=True)
        return json_lines

    @staticmethod
    def run_task_generation_with_retry(prompt):
        ai_utils = AIUtilities()

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
        def retry_run_task_generation(prompt):
            tasks = ai_utils.run_openai_completion(prompt)

            try:
                task_list = json.loads(tasks)
                valid_list = []
                for task in task_list["Tasks"]:
                    print(task["Schema"])
                    valid_list.append(task)
                return valid_list
            except Exception as e:
                print(tasks)
                print(e)

        return retry_run_task_generation(prompt)

    def create_examples(self):
        class Allergies(BaseModel):
            allergyName: str
            severity: str
            icdCodes: List[str]

        class MedicalCondition(BaseModel):
            conditionName: str
            diagnosisDate: date
            treatment: str
            icdCodes: List[str]

        class Medication(BaseModel):
            medicationName: str
            dosage: str
            frequency: str
            ndc: str
            snomed: str

        class EmergencyContact(BaseModel):
            contactName: str
            relationship: str
            phone: str
            email: EmailStr
            icdCodes: List[str] = []

        class MedicalHistory(BaseModel):
            previousSurgeries: List[str]
            vaccinations: List[str]
            familyMedicalHistory: List[str]
            icdCodes: List[str]
            cptCodes: List[str]

        examples = [
            {"Category": "JSON Schema", "SubCategory": "Healthcare System Schema", "Task": "Create a JSON object for storing patient allergies, with properties like allergyName, severity, and associated ICD codes.",
             "Schema": json.dumps(Allergies.model_json_schema())},
            {"Category": "JSON Schema", "SubCategory": "Healthcare System Schema",
             "Task": "Develop a JSON schema for representing a medical condition with properties such as conditionName, diagnosisDate, treatment, and associated ICD codes.",
             "Schema": json.dumps(MedicalCondition.model_json_schema())},
            {"Category": "JSON Schema", "SubCategory": "Healthcare System Schema",
             "Task": "Construct a JSON object representing a medication, including properties like medicationName, dosage, frequency, NDC, and associated SNOMED.",
             "Schema": json.dumps(Medication.model_json_schema())},
            {"Category": "JSON Schema", "SubCategory": "Healthcare System Schema",
             "Task": "Design a JSON object for an emergency contact profile, including properties such as contactName, relationship, phone, email, and any relevant ICD codes.",
             "Schema": json.dumps(EmergencyContact.model_json_schema())},
            {"Category": "JSON Schema", "SubCategory": "Healthcare System Schema",
             "Task": "Generate a JSON object representing a patient's medical history, including properties like previous surgeries, vaccinations, family medical history, ICD codes, and CPT codes.",
             "Schema": json.dumps(MedicalHistory.model_json_schema())},
        ]

        return examples

    def create_prompt(self, task, examples, schema):
        prompt = """
        I have a json lines task list for various GICs industries to store json schema for data entry and api calls to enterprise databases or software applications.
        The current list is missing json schema part and I need you to help add the "Schema" key to the json lines row provided below.

        Here's an example of 5 json object tasks from healthcare industry to use as an example:
        {examples}

        Here are some instructions:
        - Please return 5 diverse tasks and examples in the same GICs industry.
        - The task should not just be about creating a schema but about creating a json object adhering to the schema that assists with user query.
        - The schema should be a json schema or a pydantic schema representing data sample entry to a database or API query to a software application.
        - Please enclose all the json objects as a list with [].
        - Please create tasks and schema for the same "Category" and "SubCategory" for all 5 rows.
        - Please use this schema as a guide for each entry of the json objects list:
        {schema}

        Here's the json lines row for GICs industry that you need to return a list of 5 json objects with the task updated and the schema added:
        {task}
        """

        formatted_prompt = prompt.format(
            examples=examples,
            schema=schema,
            task=task
        )

        return formatted_prompt
    
    def process_task(self, task):
        formatted_prompt = self.create_prompt(task, self.create_examples(), TaskJsonLinesList.model_json_schema())
        print(formatted_prompt)  # Print to console

        updated_tasks = self.run_task_generation_with_retry(formatted_prompt)

        # Write to the file
        for new_task in updated_tasks:
            self.write_to_file(json.dumps(new_task))

    def run_parallel_tasks(self, json_lines_list):
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Skip the first two elements in json_lines_list
            for _ in executor.map(self.process_task, json_lines_list[5:]):
                pass

    def write_to_file(self, formatted_prompt):
        # Open the file in append mode
        with open(self.output_file_path, "a") as output_file:
            output_file.write(formatted_prompt + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tasks from GICS Classification.")
    parser.add_argument("--excel_path", type=str, help="path to GICS classification excel file")
    parser.add_argument("--output_path", type=str, help="path to output file")
    #excel_path = "/home/interstellarninja/llm_projects/data-genie/prompt_assets/curriculum/GICS Map 2023.xlsx"
    #output_file_path = "gics_curriculum.jsonl"
    args = parser.parse_args()
    task_generator = GICSTaskGenerator(args.excel_path, args.output_path)

    json_lines = task_generator.generate_json_lines()
    json_lines_list = json_lines.strip().split('\n')
    task_generator.run_parallel_tasks(json_lines_list)