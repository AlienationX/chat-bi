import gradio as gr
import requests
import json

# Ollama API 配置
OLLAMA_URL = "http://localhost:11434/api/generate"

def chat_with_deepseek(message, history):
    """与 DeepSeek 模型对话"""
    
    # 构建请求数据
    data = {
        "model": "deepseek-r1:8b",  # 替换为你的模型名称
        "prompt": message,
        "stream": False
    }
    
    try:
        # 发送请求到 Ollama
        response = requests.post(OLLAMA_URL, json=data)
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        return result["response"]
    
    except Exception as e:
        return f"错误: {str(e)}"

# 创建 Gradio 界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 DeepSeek 本地聊天机器人")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入你的问题",
                    placeholder="请输入你想问的问题...",
                    scale=4
                )
                submit_btn = gr.Button("发送", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("清空对话")
                reload_btn = gr.Button("重新加载模型")
    
    # 事件处理
    def respond(message, chat_history):
        bot_message = chat_with_deepseek(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# 启动服务
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,
        share=False  # 设置为 True 可生成公共链接
    )