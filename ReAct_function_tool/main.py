import os
import json
import random
import time
import string
import openai
from openai import AzureOpenAI

import utils
from question_loader import *
from retriever import Retriever

#OpenAI configs
client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_BASE_2"),
    api_key=os.getenv("OPENAI_API_KEY_2"),
    api_version=os.getenv("OPENAI_API_VERSION_2")
)

#Demo retrieval function

def rag(query):
    retriever = Retriever(k=5) 
    docs = retriever.retrieve(query=query)
    return docs.get_texts_as_str(token=f"\n\n\n{100*'#'}\n")

# Function to create artificial ids

def generate_id(prefix="call_", lenght=22):
    base_id = string.ascii_letters + string.digits
    id_random = ''.join(random.choices(base_id, k=lenght))
    return prefix + id_random

def llm(messages, prompt):
    """
    This function defines the API call of the llm using openai library. Then it returns the answers
    Params:
        -messages (list): list of dictionaries, each dictionary has "role" and "content" keys. First dict has the "system" prompt related with
                          the ReACT methodology. Second dict has the examples for the llm and 
        -prompt 
    Returns:
        -response_text (str): text that the llm return as the answer. It's the assistant role.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "rag",
                "description": """Use this function to retrieve information usefull for you to answer the user question or query.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": """Description of the information required to answer a question in plain text based on the user's question or query. 
                                            If the user's question or query is too complex this input should be a decomposition of the original 
                                            user question focused on a specific single piece of information.""",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-cde-aia",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    return response


def few_shots_messages_list_creator():
    """
    This fuction creates the messages list for the LLM input. It includes the instructions and the few-shot examples
    Params:
        -split (str): split of examples ('train', 'dev' or 'test')
    Returns:
        -messages(list): list of message objects, where each object has a role (either "system", "user", or "assistant") 
        and content. Based in the few-shot examples.
    """
    #Create messages list
    messages = []
    #Add first message object (role='system')
    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
    (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
    (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
    (3) Finish[answer], which returns the answer and finishes the task.
    Here are some examples.
    """
    messages.append({"role": "system", "content": instruction})

    #Load the few-shots examples file
    folder = './prompts/'
    prompt_file = 'few_shot_examples_spanish.json'
    with open(folder + prompt_file, 'r') as f:
        prompt_dict = json.load(f)

    #Examples list
    webthink_examples = prompt_dict['examples']
    breakpoint()
    #Delete 'Question' in all the examples
    webthink_examples = webthink_examples.replace("Question:", "")
    #Split all the steps in the examples and clean empty values
    webthink_examples = list(filter(lambda x: x is not None and x != "", webthink_examples.split("\n")))
    #List structure: [Question1, Thought1, Act1, Obs1, ..... , Question_n, Thought_n, Act_n, Obs_n]
    idx = 0
    
    while idx < len(webthink_examples):
        if "Thought" in webthink_examples[idx]:
            search_object =  webthink_examples[idx+1][webthink_examples[idx+1].find('[')+1: webthink_examples[idx+1].find(']')]
            #Thought and Action information
            if "Finish" in  webthink_examples[idx+1]:
                answer = {
                    'role': 'assistant',
                    'content': f"{webthink_examples[idx][11:]} ANSWER[{search_object}]"}
                messages.append(answer)
                idx += 1
            else:
                data_call = {'role': 'assistant',
                'content': webthink_examples[idx][11:],
                'function_call': None,
                'tool_calls': [{'id': generate_id(),
                                'function': {'arguments': "{\n\"query\": " + search_object + "\"\"\n}",
                                                'name': 'rag'},
                                'type': 'function'}]
                }
                messages.append(data_call)
            
                #Observation information (the response of the retrieval)
                messages.append({ "role": "tool",
                                'tool_call_id': data_call['tool_calls'][0]['id'],
                                "name": data_call['tool_calls'][0]['function']['name'],
                                "content": webthink_examples[idx+2][15:]})
                idx += 3
        else:
            if "Action" in webthink_examples[idx]:
                pass
            else:
                messages.append({"role": "user", "content": webthink_examples[idx]})
            idx += 1

    return messages

def webthink(question, messages, to_print=True):
    available_tools = {
            "rag": rag
        }
    
    if to_print:
        print(question)
    prompt = question + "\n"
    messages.append({"role": "user", "content": prompt})

    #FIXME: remenber to change the maximun times parameter to 8
    for i in range(1, 4):
        response = llm(messages, prompt)
        response_message = response.choices[0].message
        messages.append(response_message.dict())

        if 'ANSWER' in response_message:
            break

        tool_calls = response_message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                #FIXME: cases with more than one function calling not working
                function_name = tool_call.function.name
                function_to_call = available_tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Tool requested, calling function: " + str(function_name))
                function_response = function_to_call(
                    query=function_args.get("query")
                )
                messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                        })
        if to_print:
            #Thought
            print(f'Thought {i}: {response_message.content}')
            #Actions
            if tool_calls:
                for j, tool_call in enumerate(tool_calls):
                    print(f'Action {i} - {j} : {tool_call.function}')
            else:
                print('No function calling')
            #Observation
            print(f'Obs {i}: {messages[-1]['content']}')

    #Return only answer
    return messages[-1]['content']



def main():
    messages = few_shots_messages_list_creator()
    #Create QuestionLoader
    loader = QuestionLoader(split='dev')

    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    for i in idxs[:2]:
        print('--'*50)
        question = loader.load_question(idx=i)
        gt = loader.get_gt(idx=i)
        answer = webthink(question=question, messages=messages, to_print=True)
        print('Evaluation Metrics')
        print(f'Prediction: {answer}')
        print(f'Ground Truth: {gt}')
        print(get_metrics(answer, gt))



if __name__ == "__main__":
    main()
