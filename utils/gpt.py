import openai
import gradio as gr
import os 
import re

class ChatBot():
    def __init__(self, api_key):
        openai.api_key = "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"          
        self.attributes = []

    def init_chat(self): 
        messages = [ {"role": "system", "content":
                    f'''You are a helpful assistant to help user work on improving AI visual models. \
                        You need to discuss with your user for a description of the task that the model is working for. \
                        You need to decide if the description is complete and clear enough. \
                        The description should at least contains or infer the task object, task type, task scene. \
                        After understanding user's task description, you should generate related visual attributes that might affect the model's performance. You should not ask me to provide visual attributes.  \
                        (Note that this is only an example visual attributes according to the previous example, do not take any of its values as default value!):\
                        {{\
                                \"Gender , Age , Hairstyle , Hair colour\""
                        }}
                        If user is satisfied with the attributes, generate the attribute form with the header formatted as "//Attribute Form//" and end with "//END//". Attributes in the form should be splited by comma. Do not include the task object, task type, task scene.\
                        (Note that this is only an example visual attributes according to the previous example, do not take any of its values as default value!):\
                        {{\
                                \"//Attribute Form// Gender , Age , Hairstyle , Hair colour //END//\""
                        }}
                    '''
                } ]
        messages.append({"role": "assistant", "content": 'Hi, I am your assistant. Please tell me about the task of your model.'})
        messages.append({"role": "user", "content": 'I may not use English to finish the conversation, you should talk with me using my language.'})
        messages.append({"role": "assistant", "content": 'OK, I will try my best to understand you.'})
        messages.append({"role": "user", "content": 'I will let you assist me to generate a form for computer vision related tasks, you should not push me too hard.'})
        messages.append({"role": "assistant", "content": 'OK, I will not push you, and I will try to lead you to finish the form step by step.'})
        messages.append({"role": "user", "content": 'Also, you should only ask one element in one step.'})    
        messages.append({"role": "assistant", "content": 'OK, got it.'})
        messages.append({"role": "user", "content": 'You should first ask me for a description of the task.'})
        messages.append({"role": "assistant", "content": 'OK, got it.'})
        messages.append({"role": "user", "content": "Then, if you don't think the description is complete and clear enough, you should ask me about it directly!"})
        messages.append({"role": "assistant", "content": 'OK, I will ask you directly.'})
        messages.append({"role": "user", "content": "If you think the description is complete and clear enough, you should generate 4-5 visual attributes, that might affect the model's performance and related to the task object and scene."})
        messages.append({"role": "assistant", "content": 'OK, I will finally generate the form and ask if you are satisfied with the form.'})
        messages.append({"role": "user", "content": 'OK, let us start!'})
        messages.append({"role": "assistant", "content": 'Sure, now you can tell me about the task of the model now!'})
        
        return messages

    def respone(self, messages, history):
        history_format = self.init_chat()
        for human, assistant in history:
            history_format.append({"role": "user", "content": human })
            history_format.append({"role": "assistant", "content":assistant})
        history_format.append({"role": "user", "content": messages})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages = history_format, 
            temperature=1.0,
            stream=True
        )
        partial_message = ""
        for chunk in response:
            if len(chunk['choices'][0]['delta']) != 0:
                partial_message = partial_message + chunk['choices'][0]['delta']['content']
                results = self.find_attributes(partial_message)
                if len(results):
                    self.attributes = [attribute.strip() for attribute in results.split(',')]
                yield partial_message

    def find_attributes(self, content):
        start_match = re.search(r'//Attribute Form//', content)
        end_match = re.search(r'//END//', content)
        if start_match and end_match:
            start_index = start_match.end()
            end_index = end_match.start()
            attribute_list = content[start_index:end_index].strip()
        else:
            attribute_list = []
        return attribute_list

    def chat(self):
        with gr.Blocks() as demo:
            gr.ChatInterface(
                self.respone, 
                title = "Please provide a description of your task.",
                retry_btn= None,
                )
        demo.launch()

def get_des_question(main_object, attribute):
    question = """
         Give me a question asking the %s of the %s in the photo.
    """%(attribute,main_object)
    messages = [ {"role": "system", "content":"You are a intelligent assistant."} ]
    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_binary_question(main_object, attribute):
    question = """
        Give me a question asking if the %s in the photo %s or not. Return the question only. No other words. No quotation mark.
    """%(main_object,attribute)
    messages = [ {"role": "system", "content":"You are a intelligent assistant."} ]
    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_des_question(main_object, attribute):
    question = """
         Give me a question asking the %s of the %s in the photo. Return the question only. No other words. No quotation mark.
    """%(attribute,main_object)
    messages = [ {"role": "system", "content":"You are a intelligent assistant."} ]
    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_des_prompt(main_object, attribute):
    question = """
        Give me a sentence combining a visual feature and a main object. For example, when the feature is hair color, and the main object is people. 
        The sentence could be: A #LABEL with #1 hair. Here #LABEL is related to the main object. #1 represent a specific type of the visual feature. Your sentence also includes the #LABEL and #1. 
        Do not include other things in the sentence. It should be as simple as possible. Now the visual feature is %s, and the object is %s. Give me a sentence. No other words. No quotation mark.
    """%(attribute,main_object)
    messages = [ {"role": "system", "content":"You are a intelligent assistant."} ]
    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_type(main_object, attribute):
    question = """
        When we ask about %s of %s, is it binary (yes, no) or description (red, blue, yellow for color)? You only need to tell me binary or description. No punctuation.
    """%(attribute,main_object)
    messages = [ {"role": "system", "content":"You are a intelligent assistant."} ]
    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply