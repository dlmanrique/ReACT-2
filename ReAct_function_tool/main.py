import os
import json
import random
import time
import string
import openai
from openai import AzureOpenAI
import copy

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
retriever = Retriever() 
def rag(query, retriever=retriever):
    docs = retriever.retrieve(query=query, table_name= 'gestion_humana')
    return docs.get_texts_as_str(token=f"\n\n\n{100*'#'}\n")

# Function to create artificial ids
def generate_id(prefix="call_", lenght=22):
    base_id = string.ascii_letters + string.digits
    id_random = ''.join(random.choices(base_id, k=lenght))
    return prefix + id_random

def llm(messages):
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
                "description": """Utilice esta función para recuperar información útil para responder a la pregunta o consulta del usuario.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": """Descripción de la información necesaria para responder a una pregunta en texto plano basada en la pregunta o consulta del usuario. 
                                            Si la pregunta o consulta del usuario es demasiado compleja, esta entrada debe ser una descomposición de la pregunta original del usuario 
                                            centrada en un único dato específico de la pregunta original del usuario centrada en una única información específica.""",
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
    instruction = """Resolve una pregunta intercalando pasos de Pensamiento, Acción y Observacion.
      El Pensamiento consiste en razonar y reflexionar sobre la situación actual, y la Accion consiste en realizar consultas de 
      palabras clave para obtener la informacion necesaria para responder la pregunta. La observacion es analizar la informacion 
      obtenida en la accion inmediatamente anterior. Cuando obtengas la respuesta a la pregunta, no debe ser en la primer fase
      de reflexion, dejala en el siguiente formato: ANSWER[respuesta a la pregunta]. Aqui te muestro unos ejemplos de esta metodologia:
    """
    messages.append({"role": "system", "content": instruction})

    #Load the few-shots examples file
    folder = './prompts/'
    prompt_file = 'few_shot_examples_spanish.json'
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

def webthink(question, example_messages, idx, available_tools, to_print=True, record= False):
    start_time = time.time()
    messages = copy.deepcopy(example_messages)
    
    
    if to_print:
        print(question)
    prompt = question + "\n"
    original_lenght = len(messages)
    messages.append({"role": "user", "content": prompt})
    react_iterations = 0
    for i in range(1, 8):
        react_iterations += 1
        response = llm(messages)
        response_message = response.choices[0].message
        messages.append(response_message.dict())
        
        #Case when it just call function and not make the 'thought'
        if response_message.content is not None and 'ANSWER' in response_message.content:
            break       

        tool_calls = response_message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Tool requested, calling function: {function_name} \n Query: {function_args.get("query")}")
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
            print(f'Pensamiento {i}: {response_message.content}')
            #Actions
            if tool_calls:
                for j, tool_call in enumerate(tool_calls):
                    print(f'Accion {i} - {j} : {tool_call.function}')
            else:
                print('No function calling')
            #Observation
            print(f'Observacion {i}: {messages[-1]['content']}')

    question_messages_info = messages[original_lenght:]
    elapsed_time = time.time() - start_time
    if record == True:
        with open(f'experiment_longer_few_shots/question_info{idx}.json', 'w', encoding='utf-8') as json_file:
            json_info_dict = {'react_iterations ': react_iterations, 
                              'time consumed': elapsed_time,
                              'messages': question_messages_info}
            json.dump(json_info_dict, json_file, ensure_ascii=False, indent=4)
    #Return only answer
    return messages[-1]['content']



def main():
    #example_messages = few_shots_messages_list_creator()
    with open('few_shot_examples_extended_answers.json', 'r', encoding='utf-8') as archivo:
        # Paso 3: Usar json.load() para leer el archivo
        example_messages = json.load(archivo)
    #breakpoint()
    #Available tools
    available_tools = {
            "rag": rag
        }
    #Create QuestionLoader
    loader = QuestionLoader()
    all_questions_info = []
    for i in range(4, 5):
        if i % 10 == 0 and i > 0:
            print("Pausando ejecución...")
            time.sleep(30)  # Pausa por 30 segundos
            print("Reanudando ejecución.")
        print('--'*70)
        question = loader.load_question(idx=i)
        gt = loader.get_gt(idx=i)
        answer = webthink(question, example_messages, idx = i+1, available_tools= available_tools, to_print=True, record=True)
        #breakpoint()
        print('Evaluation Metrics')
        print(f'Prediction: {answer}')
        print(f'Ground Truth: {gt}')

        match = re.search(r'ANSWER\[(.*?)\]', answer)
        if match:
            answer = match.group(1)
        question_info = {'unique_id': i+1, 'question': question, 'qs_response': answer, 'ground_truth': gt}
        all_questions_info.append(question_info)
    
    with open(f'React_GH_answers.json', 'w', encoding='utf-8') as json_file:
            json.dump(all_questions_info, json_file, ensure_ascii=False, indent=4)
        



if __name__ == "__main__":
    main()
