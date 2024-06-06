import os
from dotenv import load_dotenv
from schema import OutputSchema

#import together
from openai import OpenAI, AzureOpenAI
from groq import Groq
#from anthropic import Anthropic

class AIUtilities:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        
        self.openai_key = os.getenv("OPENAI_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")

        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        self.together_model = os.getenv("TOGETHER_MODEL")

        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.api_version = os.getenv("API_VERSION")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_model = os.getenv("AZURE_OPENAI_MODEL")

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL")

        self.anyscale_api_key = os.getenv("ANYSCALE_API_KEY")
        self.anyscale_model = os.getenv("ANYSCALE_MODEL")

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL")

        self.ouput_schema = OutputSchema.schema_json()

    def run_ai_completion(self, prompt, ai_vendor):
        if ai_vendor == "openai":
            return self.run_openai_completion(prompt)
        if ai_vendor == "azure_openai":
            return self.run_azure_openai_completion(prompt)
        elif ai_vendor == "anthropic":
            return self.run_anthropic_completion(prompt)
        elif ai_vendor == "together":
            return self.run_together_completion(prompt)
        elif ai_vendor == "anyscale":
            return self.run_anyscale_completion(prompt)
        elif ai_vendor == "groq":
            return self.run_groq_completion(prompt)
        else:
            return "Invalid AI vendor"
    
    def get_ai_context_length(self, ai_vendor):
        if ai_vendor == "openai":
            return os.getenv("OPENAI_CONTEXT_LENGTH")
        if ai_vendor == "azure_openai":
            return os.getenv("AZURE_OPENAI_CONTEXT_LENGTH")
        elif ai_vendor == "anthropic":
            return os.getenv("ANTHROPIC_CONTEXT_LENGTH")
        elif ai_vendor == "groq":
            return os.getenv("GROQ_CONTEXT_LENGTH")
        elif ai_vendor == "together":
            return os.getenv("TOGETHER_CONTEXT_LENGTH")
        elif ai_vendor == "anyscale":
            return os.getenv("ANYSCALE_CONTEXT_LENGTH")
        else:
            return "Invalid AI vendor"

        
    def run_openai_completion(self, prompt):
        client = OpenAI(
            api_key=self.openai_key
        )
        try:
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)
        
    def run_ai_tool_completion(self, messages, tools, tool_choice="auto", json=False):
        client = OpenAI(
            api_key=self.openai_key,
        )
        if json:
            response_format = {"type": "json_object"}
        else:
            response_format = None
        try:    
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format
            )
            completion = response.choices[0].message
            return completion
        except Exception as e:
            return str(e)

    def run_azure_openai_completion(self, prompt):
        client = AzureOpenAI(
            api_key=self.azure_openai_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_openai_endpoint
        )
        try:
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0,
                frequency_penalty=1,
                presence_penalty=0.5
            )

            # Extract and save results
            completion = response.choices[0].message.content
            return completion
        except Exception as e:
            return str(e)

    def run_anthropic_completion(self, prompt):
        anthropic = Anthropic(api_key=self.anthropic_api_key)
        try:
            response = anthropic.beta.messages.create(
                model="claude-2.1",
                max_tokens=1024,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                    #{"role": "assistant", "content": "Here's a json object:\n"},
                ]
            )
            #response = anthropic.completions.create(
            #    prompt=prompt,
            #    stop_sequences=["\n\nHuman:", "</answer>"],
            #    model=self.anthropic_model,
            #    max_tokens_to_sample=1024,
            #    temperature=0
            #)
            return response.completion
        except Exception as e:
            return str(e)
        
    def run_together_completion(self, prompt):
        together.api_key = self.together_api_key
        try:
            response = together.Complete.create(
                model=self.together_model,
                prompt=prompt,
                max_tokens=1024,
                temperature=0,
                repetition_penalty=1.5
            )
            return response["output"]["choices"][0]['text']
        except Exception as e:
            return str(e)
        
    def run_anyscale_completion(self, prompt):
        client = OpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key = self.anyscale_api_key)

        try:
            chat_completion = client.chat.completions.create(
                model=self.anyscale_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs in JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_object",
                    "schema": self.output_schema,
                },
                max_tokens=1024,
                temperature=0
            )
            response = chat_completion.model_dump()
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return str(e)
        
    def run_groq_completion(self, prompt):
        try:
            client = Groq(
                # This is the default and can be omitted
                api_key=self.groq_api_key,
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.groq_model,
            )
            print(chat_completion)
            return chat_completion.choices[0].message.content
        except Exception as e:
            return str(e)

def main():
    load_dotenv()  # Load environment variables from .env file

    ai_utilities = AIUtilities()

    # Example usage
    prompt = "Tell me a programmer joke"
    ai_vendor = "openai"  # Change this to the desired AI vendor

    # Run AI completion
    result = ai_utilities.run_ai_completion(prompt, ai_vendor)
    print(f"AI Completion Result:\n{result}")

if __name__ == "__main__":
    main()