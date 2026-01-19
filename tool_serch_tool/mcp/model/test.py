from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    api_key="not-needed",
    temperature=0,
)



response = model.invoke('tell A JOKE')
print(response.content)