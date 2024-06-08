import json
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field
from datetime import datetime
from src.clients import CLIENTS
from src import tools
from src.tools import *
from src.rag_tools import *
from src.utils import inference_logger
from src.utils import validate_and_extract_tool_calls
from langchain.tools import StructuredTool, BaseTool

from langchain_core.messages import ToolMessage

## TODO: add default tools such as "get_user_feedback", "get_additional_context", "code_interpreter" etc.

class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    system: str
    tools: List[str] = []
    resources: List[str] = []
    dependencies: Optional[List[str]] = None
    llm_config: Dict = Field(default_factory=dict)
    max_iter: int = 5
    input_messages: List[Dict] = []
    interactions: List[Dict] = []
    verbose: bool = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.config_path = config_path
        self.prompt_path = prompt_path
        self.model_config = self.load_model_config(config_path)
        self.system_prompt = self.load_system_prompt(prompt_path)
        if not self.client:
            raise ValueError("Invalid client specified.")
        self.tool_objects = self.create_tool_objects()

    def execute(self) -> str:
        messages = []
        if self.input_messages:
            for input_message in self.input_messages:
                role = input_message["role"]
                content = input_message["content"]
                inference_logger.info(f"Appending input messages from previous agent: {role}")
                messages.append({"role": "system", "content": f"<agent_messages>\n<{role}>\n{content}\n</{role}>\n</agent_messages>"})
        
        if self.name and self.verbose:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            system_prefix = f"You are a {self.name} AI agent. Current date and time: {current_datetime}"

        prompt_manager = PromptManager(self.config)
        prompt_yaml_path = os.path.join(self.config["paths"]["prompt_yaml"], f"{self.generation_type}.yaml")
        
        prompt_messages = prompt_manager.generate_prompt_messages(variables, prompt_yaml_path, system_prefix=system_prefix)
        logger.info(f"Logging prompt text\n{messages}")

        messages.extend(prompt_messages)

        depth = 0
        while depth < self.max_iter:
            inference_logger.info(f"Running inference with {self.client}")
            ai_utilities = AIUtilities()
            completion = ai_utilities.run_ai_completion(prompt, self.llm_config)
            logger.info(f"Here's the generated json output:\n{completion}")
            # Extract and save results for each task
            logger.info(f"saving json files for {task_desc}")
            self.extract_and_save_results(file_path, completion, task_desc)
            
            inference_logger.info(f"Assistant Message:\n{completion}")
            messages.append({"role": "assistant", "content": completion})

            # Process the agent's response and extract tool calls
            if self.tools:
                pass
        # Log the final interaction
        self.log_interaction(messages, result)
        return result

    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(tools, function_name, None)
        function_args = tool_call.get("arguments", {})

        if function_to_call:
            inference_logger.info(f"Invoking function call {function_name} ...")
            function_response = function_to_call(**function_args)
            results_dict = f'{{"name": "{function_name}", "content": {json.dumps(function_response)}}}'
            return results_dict
        else:
            raise ValueError(f"Function '{function_name}' not found.")

    def log_interaction(self, prompt, response):
        self.interactions.append({
            "role": self.role,
            "messages": prompt,
            "response": response,
            "agent_messages": self.input_messages,
            "tools": self.tools,
            "timestamp": datetime.now().isoformat()
        })