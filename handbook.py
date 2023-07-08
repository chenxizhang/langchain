def get_predit_llm_huggingface(prompt:str):
    from langchain import HuggingFaceHub, LLMChain, PromptTemplate
    hub_llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 1e-10}) # type: ignore
    llm_chain = LLMChain(llm=hub_llm, prompt=PromptTemplate.from_template(prompt))
    print(llm_chain)
    print(llm_chain.predict())


if __name__ == "__main__":
    get_predit_llm_huggingface("What is the capital of China?")