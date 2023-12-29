from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class UserMessage(BaseModel):
    role: str = "user"
    content: str = Field(
        "The user's query to assist with a task by calling a function. Include any additional context regarding the task. Provide data such as documents, tables, files, code, documentation, etc. that the function call may require."
    )

class FunctionCall(BaseModel):
    name: str
    arguments: Optional[Union[Dict[str, Any], str]] = Field(
        default=None,
        description="The parameters for the function call. It could be a dictionary of key-value pairs or a string, depending on the function."
    )

class AssistantMessageToolCall(BaseModel):
    id: str
    function: FunctionCall
    type: str
    
class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = Field(
        default=None,
        description="If the assistant message is followed by a user role and a function_call is provided, this field will be null. If the assistant message is followed by a function role, it will be a summary of the function call results."
    )
    #function_call: Optional[FunctionCall] = Field(
    #    default=None,
    #    description="The 'function_call' property includes 'name' (the name of the function to be called) and 'arguments' (the parameters for the function call)."
    #)
    tool_calls: Optional[List[AssistantMessageToolCall]] = Field(
        default=None,
        description="The 'tool_calls' property includes a list of function calls providing information about the arguments of functions called by the assistant."
    )

class ToolMessage(BaseModel):
    role: str = "tool"
    tool_call_id: str
    name: str
    content: Optional[FunctionCall] = Field(
        default=None,
        description="The content of the tool message, which is a function call with 'name' (the name of the function to be called) and 'arguments' (the parameters for the function call)."
    )

class ParameterProperty(BaseModel):
    type: str
    description: Optional[str]
    enum: Optional[Dict[str, str]]

class Parameter(BaseModel):
    type: str
    properties: Dict[str, ParameterProperty]
    required: List[str]

class FunctionSignature(BaseModel):
    name: str
    description: str
    parameters: Parameter

class OutputSchema(BaseModel):
    messages: List[Union[UserMessage, AssistantMessage, ToolMessage]] = Field(
        description="The messages array contains the chain of messages between the user, assistant, and function to assist with the user query."
    )
    tools: List[FunctionSignature] = Field(
        description="The tools array contains information about available functions or tools that can be called to answer the user query."
    )
