import json
from pydantic import BaseModel, ValidationError, validator, create_model
from typing import List, Dict, Literal, Optional
from jsonschema import validate, ValidationError

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, object]] = None

class FunctionSignature(BaseModel):
    function: FunctionDefinition
    type: Literal["function"]

class FunctionCall(BaseModel):
    arguments: dict
    name: str

class FunctionCallMessage(BaseModel):
    id: str
    function: FunctionCall
    type: Literal["function"]

def validate_function_call(call, signatures):
    try:
        call_name = call.name
        call_arguments = call.arguments
    except ValidationError as e:
        return False, {'status': 'failed', 'message': f"Invalid function call: {e}", 'signature': None, 'call': call.dict()}

    for signature in signatures:
        # Inside the main validation function
        try:
            if signature.function.name == call_name:
                signature_data = FunctionSignature(**signature.dict())
                # validate_signature_fields(FunctionSignature, signature.function, ['name', 'description', 'parameters'])

                # Validate types in function arguments
                for arg_name, arg_schema in signature_data.function.parameters.get('properties', {}).items():
                    if arg_name in call_arguments:
                        call_arg_value = call_arguments[arg_name]
                        if call_arg_value:
                            try:
                                validate_argument_type(arg_name, call_arg_value, arg_schema)
                            except Exception as arg_validation_error:
                                return False, {'status': 'failed',
                                               'message': f"Invalid argument '{arg_name}': {arg_validation_error}",
                                               'call': call.dict(), 'signature': signature.dict()}

                # Check if all required arguments are present
                required_arguments = signature_data.function.parameters.get('required', [])
                result, missing_arguments = check_required_arguments(call_arguments, required_arguments)

                if not result:
                    return False, {'status': 'failed',
                                   'message': f"Missing required arguments: {missing_arguments}",
                                   'signature': signature.dict(), 'call': call.dict()}

                return True, {'status': 'accepted', 'message': "Function call is valid",
                              'signature': signature.dict(), 'call': call.dict()}
        except Exception as e:
            # Handle validation errors for the function signature
            return False, {'status': 'failed', 'message': f"Error validationg function call: {e}", 'signature': signature.dict(), 'call': call.dict()}

    # Moved the "No matching function signature found" message here
    return False, {'status': 'failed', 'message': f"No matching function signature found for function: {call_name}",
                   'signature': None, 'call': call.dict()}


def check_required_arguments(call_arguments, required_arguments):
    missing_arguments = [arg for arg in required_arguments if arg not in call_arguments]
    return not bool(missing_arguments), missing_arguments

def validate_enum_value(arg_name, arg_value, enum_values):
    if arg_value not in enum_values:
        print(enum_values)
        raise Exception(
            f"Invalid value '{arg_value}' for parameter {arg_name}. Expected one of {', '.join(map(str, enum_values))}"
        )

def validate_argument_type(arg_name, arg_value, arg_schema):
    arg_type = arg_schema.get('type', None)
    if arg_type:
        if arg_type == 'string' and 'enum' in arg_schema:
            enum_values = arg_schema['enum']
            if None not in enum_values and enum_values != []:
                try:
                    validate_enum_value(arg_name, arg_value, enum_values)
                except Exception as e:
                    # Propagate the validation error message
                    raise Exception(f"Error validating function call: {e}")

        python_type = get_python_type(arg_type)
        if not isinstance(arg_value, python_type):
            raise Exception(f"Type mismatch for parameter {arg_name}. Expected: {arg_type}, Got: {type(arg_value)}")


def validate_signature_fields(model, values, required_fields):
    print(values)
    for field in required_fields:
        if field not in values:
            print(field)
            raise Exception(f"Function signature missing required field: {field}")

def get_python_type(json_type):
    type_mapping = {
        'string': str,
        'number': (int, float),
        'integer': int,
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': type(None),
    }
    return type_mapping[json_type]

def validate_function_calls(tool_calls, function_signatures):
    failed_flag = False
    results = []

    for call_data in tool_calls:
        try:
            # Convert "arguments" from JSON string to dictionary
            call_data["function"]["arguments"] = json.loads(call_data["function"]["arguments"])
            # Load the function call message as a FunctionCallMessage Pydantic object
            function_call_message = FunctionCallMessage(**call_data)                        
            # Extract the function call from the message
            function_call = function_call_message.function
            # Validate the function call
            result, result_data = validate_function_call(function_call, function_signatures)
            results.append(result_data)
            if result is False:
                failed_flag = True
        except Exception as e:
            # Handle validation errors
            results.append({'status': 'failed',
                            'message': f"Invalid tool call: {e}",
                            'call': function_call.dict()})
            failed_flag = True
    return results, failed_flag

def validate_json_object(data, schema):

    # Pre-process data
    data = json.loads(json.dumps(data))

    # Pre-process schema
    schema = json.loads(json.dumps(schema))
    
    # Create Pydantic models and validate
    schema_model = create_pydantic_model(schema)
    try: 
        schema_model(**data)
    except ValidationError as e:
        print(f"Validation Error: {e}")
        return False

    return True

def create_pydantic_model(schema_dict, base_name="Model"):
    fields = {}
    for name, prop in schema_dict.items():
        if "$ref" in prop:
            ref_model_name = prop["$ref"].split("/")[-1]
            fields[name] = (create_model(schema_dict[ref_model_name], base_name=ref_model_name),)
        else:
            fields[name] = (prop["type"],)
    
    model_name = f"{base_name}Model"
    model = type(model_name, (BaseModel,), fields)
    return model

def validate_json_data(json_object, json_schema):
    valid = True
    
    # Validate each item in the list against schema if it's a list
    if isinstance(json_object, list):
        for index, item in enumerate(json_object):
            try:
                validate(instance=item, schema=json_schema)
                print(f"Item {index+1} is valid against the schema.")
            except ValidationError as e:
                valid = False
                print(f"Validation failed for item {index+1}: {e}")
    else:  # Default to validation without list
        try:
            validate(instance=json_object, schema=json_schema)
            print("JSON object is valid against the schema.")
        except ValidationError as e:
            print("Validation failed:", e)
            valid = False

    if valid:
        print("JSON data is valid against the schema.")
    else:
        print("Validation failed for JSON data.")
    return valid




