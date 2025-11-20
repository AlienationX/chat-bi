from langchain.agents import create_agent
from langchain_ollama import ChatOllama


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print(">>> tool message: get_weather", city)
    return f"It's always sunny in {city}!"


llm = ChatOllama(
    # qwen2.5:1.5b, deepseek-v3.1:671b-cloud, deepseek-r1:8b
    model="qwen2.5:1.5b",  # 选择任何你本地已有的模型,deepseek-r1:8b不支持工具调用
    temperature=0.7,  # 控制生成结果的随机性
    num_ctx=4096,  # 设置上下文窗口大小
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})

print(result["messages"][-1].content)
print(result)

for msg in result["messages"]:
    msg.pretty_print()


result = agent.invoke({"messages": [{"role": "user", "content": "你是谁"}]})

print(result["messages"][-1].content)
