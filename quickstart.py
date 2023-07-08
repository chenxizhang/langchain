def get_predict_llm(prompt:str):
    # 根据提示词完成句子
    from langchain.llms import AzureOpenAI
    llm = AzureOpenAI(deployment_name="text-davinci-003") # type: ignore
    print(llm.predict(prompt))

def get_predict_chat(message:str):
    # 根据对话历史预测下一句话
    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema import (HumanMessage, SystemMessage)
    chat = AzureChatOpenAI(deployment_name="gpt-4", openai_api_version="2023-03-15-preview") # type: ignore
    print(chat.predict_messages([
        SystemMessage(content="You are a network expert, help people to solve their network problems."),
        HumanMessage(content= message)
    ]))

def get_predict_llm_template(country:str):
    # 根据模板生成句子
    from langchain.llms import AzureOpenAI
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template ("What is the capital of {country}?")
    llm = AzureOpenAI(deployment_name="text-davinci-003") # type: ignore
    print(llm.predict(prompt.format(country=country)))


def get_predict_chat_template(message:str):
    # 根据模板生成对话
    from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
    from langchain.chat_models import AzureChatOpenAI

    template = 'You are a helpful assistant that translates {input_language} to {output_language}. \n'
    system_message_template = SystemMessagePromptTemplate.from_template(template)
    human_template = '{text}'
    human_message_template = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])
    chat = AzureChatOpenAI(deployment_name="gpt-4", openai_api_version="2023-03-15-preview") # type: ignore
    print(chat.predict_messages(chat_prompt.format_messages(input_language="English", output_language="French", text=message)))

def get_predict_llm_chain(prompt:str):
    # 根据提示词完成句子
    from langchain.llms import AzureOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    llm = AzureOpenAI(deployment_name="text-davinci-003") # type: ignore
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    print(chain.predict())

def get_result_from_agent(message:str):
    # 多个任务一起执行, 自动推理和编排任务
    from langchain.agents import AgentType, initialize_agent, load_tools
    from langchain.llms import AzureOpenAI

    llm = AzureOpenAI(deployment_name="text-davinci-003") # type: ignore
    tools = load_tools(['serpapi','llm-math'],llm=llm)
    agent = initialize_agent(tools,llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run(message)

def get_predict_llm_conversation(prompt:str):
    # memory based conversation
    from langchain.llms import AzureOpenAI
    from langchain import ConversationChain
    llm = AzureOpenAI(deployment_name="text-davinci-003") # type: ignore
    conversation = ConversationChain(llm=llm, verbose=True)
    print(conversation.run(prompt))


if __name__ == "__main__":
    # get_predict_llm("Hello, my dog is cute")
    # get_predict_chat("How can I measure the speed of my internet connection?")
    # get_predict_llm_template("France")
    # get_predict_chat_template("How can I measure the speed of my internet connection?")
    # get_predict_llm_chain("Hello, my dog is cute")
    # get_result_from_agent("How far from Shanghai to London? What's the number if we double the distiance?")
    get_predict_llm_conversation("Hello, my dog is cute")
    get_predict_llm_conversation("How about your dog?")