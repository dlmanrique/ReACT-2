import os
import json
import random
import time
import string
import wikienv, wrappers
import openai
from openai.util import convert_to_openai_object

import utils

#OpenAI configs
openai.api_type = os.getenv("OPENAI_API_TYPE_2")
openai.api_key = os.getenv("OPENAI_API_KEY_2")
openai.api_base = os.getenv("OPENAI_API_BASE_2")
openai.api_version = os.getenv("OPENAI_API_VERSION_2")

#Demo retrieval function

def demo_retireval(query:str):
   return f'The information about {query} is: '

# Function to create artificial ids

def generate_id(prefix="call_", lenght=22):
    base_id = string.ascii_letters + string.digits
    id_random = ''.join(random.choices(base_id, k=lenght))
    return prefix + id_random

def llm(messages, prompt, stop=["\n"]):
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
                "name": "demo_retireval",
                "description": "Retrieve information about something",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query that I want to retrieve information",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    breakpoint()
    messages[-1]["content"] = prompt
    
    response = openai.ChatCompletion.create(
      engine="gpt-35-turbo-16k-cde-aia",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_text = response["choices"][0]["message"]["content"]
    return response_text


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            print("-"*30)
            attempts += 1

def webthink(question, messages, env, to_print=True):
    if to_print:
        print(question)
    prompt = question + "\n"
    messages.append({"role": "user", "content": prompt})
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(messages, prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(messages, prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        messages[-1]["content"] = prompt
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info


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
    prompt_file = 'few_shot_examples.json'
    with open(folder + prompt_file, 'r') as f:
        prompt_dict = json.load(f)

    #Examples list
    webthink_examples = prompt_dict['examples']
    #Delete 'Question' in all the examples
    webthink_examples = webthink_examples.replace("Question:", "")
    #Split all the steps in the examples and clean empty values
    webthink_examples = list(filter(lambda x: x is not None and x != "", webthink_examples.split("\n")))
    #List structure: [Question1, Thought1, Act1, Obs1, ..... , Question_n, Thought_n, Act_n, Obs_n]
    idx = 0

    while idx < len(webthink_examples):
        if "Thought" in webthink_examples[idx]:
            messages.append({"role": "assistant", "content": webthink_examples[idx][11:]})
            idx += 1
        elif "Action" in webthink_examples[idx]:
            search_object =   webthink_examples[idx][webthink_examples[idx].find('[')+1: webthink_examples[idx].find(']')]
            data = {
                    "function": {
                        "arguments": "{\n\"query\": " + search_object + "\"\"\n}",
                        "name": "demo_retireval"
                    },
                    "id": generate_id(),
                    "type": "function"
                }
            openai_object = convert_to_openai_object(data)
            messages.append(openai_object)
            idx += 1
        elif "Observation" in webthink_examples[idx]:
            messages.append({"role": "tool", "content": webthink_examples[idx][15:]})
            idx += 1

        else:
            messages.append({"role": "user", "content": webthink_examples[idx]})
            idx += 1

    return messages


def main():
    messages = few_shots_messages_list_creator()
    #Create env and prepare search agent
    env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(env, split='dev')
    env = wrappers.LoggingWrapper(env)

    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()
    for i in idxs[:2]:
        question = env.reset(idx=i).replace('Question: ', '')
        r, info = webthink(question=question, messages=messages, env=env, to_print=True)
        rs.append(info['em'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print()
if __name__ == "__main__":
    main()
