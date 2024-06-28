import os
import json
#from openai import AzureOpenAI
import openai

"""client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_BASE_2"),
    api_key=os.getenv("OPENAI_API_KEY_2"),
    api_version=os.getenv("OPENAI_API_VERSION_2")
)"""

openai.api_type = os.getenv("OPENAI_API_TYPE_2")
openai.api_key = os.getenv("OPENAI_API_KEY_2")
openai.api_base = os.getenv("OPENAI_API_BASE_2")
openai.api_version = os.getenv("OPENAI_API_VERSION_2")

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = openai.ChatCompletion.create(
        #model="gpt-4o-cde-aia",
        engine="gpt-35-turbo-16k-cde-aia",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit, can be required if it must call one or more tools vía {"type": "function", "function": {"name": "my_function"}}, None if no calls are needed
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response, #'{"location": "San Francisco", "temperature": "72", "unit": null}'
                }
            )  # extend conversation with function response

        """[{'role': 'user', 'content': "What's the weather like in San Francisco, Tokyo, and Paris?"}, 
        ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[
        ChatCompletionMessageToolCall(id='call_NRKC1Hd6EW0Ij2t6jSh5U7Ax', function=Function(arguments='{"location": "San Francisco, CA"}', name='get_current_weather'), type='function'), 
        ChatCompletionMessageToolCall(id='call_b8EEIBelFppb0bKMA9xlrlV0', function=Function(arguments='{"location": "Tokyo, Japan"}', name='get_current_weather'), type='function'), 
        ChatCompletionMessageToolCall(id='call_MDlCwfkCA5Swo9LnSlsaYlWC', function=Function(arguments='{"location": "Paris, France"}', name='get_current_weather'), type='function')]), 
        {'tool_call_id': 'call_NRKC1Hd6EW0Ij2t6jSh5U7Ax', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"location": "San Francisco", "temperature": "72", "unit": null}'}, 
        {'tool_call_id': 'call_b8EEIBelFppb0bKMA9xlrlV0', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"location": "Tokyo", "temperature": "10", "unit": null}'}, 
        {'tool_call_id': 'call_MDlCwfkCA5Swo9LnSlsaYlWC', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"location": "Paris", "temperature": "22", "unit": null}'}]
        """
        second_response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k-cde-aia",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response
    
print(run_conversation()) 