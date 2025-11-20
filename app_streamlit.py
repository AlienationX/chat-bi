import streamlit as st
import requests
import json
import time
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="DeepSeek Chat",
    page_icon="❉",
    layout="wide"
)

# 隐藏右上角 Deploy 按钮并自定义聊天样式
st.markdown(
    """
    <style>
        /* 仅隐藏右上角 Deploy 按钮，保留其他工具栏元素 */
        .stAppDeployButton, .stStatusWidget {
            display: none !important;
        }
        /* 用户气泡靠右显示：倒置 flex 排列实现头像与内容右侧对齐 */
        [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse !important;
            justify-content: flex-end !important;
            text-align: right !important;
        }
        [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            align-items: flex-end !important;
        }
        /* 聊天正文整体字体缩小一号，保持紧凑 */
        [data-testid="stChatMessageContent"] {
            font-size: 0.90em;
        }
        /* 思考（blockquote）区域：减小字体、控制段前后距与行高防止空白过大 */
        [data-testid="stChatMessage"] blockquote {
            font-size: 0.85em;
            color: #5a5a5a;
            border-left: 3px solid #d0d0d0;
            margin: 0.1rem 0 0.8rem 0;
            padding-left: 0.4rem;
            line-height: 1.2;
        }
        /* 普通段落：设置更小的段前/段后间距与行高 1.8，兼顾可读性与紧凑度 */
        [data-testid="stChatMessageContent"] p {
            margin: 0.15rem 0 0.9rem 0;
            line-height: 1.5;
        }
    </style>
    <script>
        // 自动滚动到底部，确保新消息可见
        function scrollToBottom() {
            // 滚动到聊天输入框
            const chatInput = document.querySelector('[data-testid="stChatInput"]');
            if (chatInput) {
                chatInput.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
            // 或者滚动到页面底部
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
        // 页面加载完成后滚动到底部
        window.addEventListener('load', scrollToBottom);
        // 使用 MutationObserver 监听 DOM 变化，当有新消息时自动滚动
        const observer = new MutationObserver(() => {
            setTimeout(scrollToBottom, 100);
        });
        // 观察主内容区域的变化
        observer.observe(document.body, { childList: true, subtree: true });
        // Streamlit 特有的：监听 Streamlit 事件
        if (window.parent !== window) {
            window.parent.addEventListener('message', (event) => {
                if (event.data.type === 'streamlit:render') {
                    setTimeout(scrollToBottom, 200);
                }
            });
        }
    </script>
    """,
    unsafe_allow_html=True,
)


def format_timestamp(ts: datetime | None = None) -> str:
    return (ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def format_thinking_markdown(raw_text: str) -> str:
    if not raw_text or not raw_text.strip():
        return ""
    lines = raw_text.split("\n")
    formatted_lines = []
    for line in lines:
        if line:
            formatted_lines.append(f"> {line.rstrip()}")
        else:
            formatted_lines.append(">")
    return "\n".join(formatted_lines).strip()


def build_conversation_prompt(
    history: list[dict], latest_prompt: str, max_turns: int = 6
) -> str:
    """
    将最近 max_turns 轮对话整理为指令式文本，帮助模型保留上下文，
    并以 'Assistant:' 结尾提示模型继续生成。
    """
    if not history:
        return f"User: {latest_prompt.strip()}\n\nAssistant:"

    trimmed_history = history[-max_turns * 2 :]
    formatted_segments: list[str] = []
    for record in trimmed_history:
        role_label = "User" if record.get("role") == "user" else "Assistant"
        content = record.get("content", "").strip()
        if content:
            formatted_segments.append(f"{role_label}: {content}")

    if trimmed_history[-1].get("role") != "user":
        formatted_segments.append(f"User: {latest_prompt.strip()}")

    formatted_segments.append("Assistant:")
    return "\n\n".join(formatted_segments)


# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "history_panel_open" not in st.session_state:
    st.session_state.history_panel_open = False

# 获取Ollama模型列表
def get_ollama_models(ollama_url):
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            if "models" in models_data:
                return [model["name"] for model in models_data["models"]]
    except Exception as e:
        st.warning(f"无法获取模型列表: {str(e)}")
    return []
    
# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 设置")
    
    # 设置为只读显示
    ollama_url = "http://localhost:11434"
    # 显示固定的 Ollama API 地址
    st.markdown(f"**Ollama API 地址:** {ollama_url}")
    
    # 获取并显示模型列表下拉选择
    models = get_ollama_models(ollama_url)
    model_name = st.selectbox(
        "选择模型",
        options=models,
        index=0,
        help="从已下载的Ollama模型中选择"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="控制回答的随机性"
    )

    st.divider()

    if st.button("查看历史记录" if not st.session_state.history_panel_open else "隐藏历史记录"):
        st.session_state.history_panel_open = not st.session_state.history_panel_open
        st.rerun()

    if st.button("清空对话记录"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("检查模型状态"):
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                st.success("模型服务运行正常")
            else:
                st.error("模型服务异常")
        except requests.RequestException as exc:
            st.error(f"无法连接到模型服务: {exc}")

# 主界面
st.title("DeepSeek 本地聊天")
st.caption("与本地部署的 DeepSeek 模型对话")

# 显示聊天记录
for message in st.session_state.messages:
    role = message["role"]
    timestamp = message.get("timestamp")
    duration = message.get("duration")
    with st.chat_message(role):
        # 回放历史消息的 Markdown 内容
        st.markdown(message["content"])
        if role == "assistant":
            st.caption(f"history = {timestamp} spend: {duration:.1f} s")
        else:
            st.caption(f"history = {timestamp}")


# 聊天输入
# if prompt := st.chat_input("请输入你的问题...", accept_file=True, file_type=["jpg", "jpeg", "png", "docx", "doc", "pdf", "xlsx", "xls", "csv", "txt"]):
if prompt := st.chat_input("请输入你的问题..."):
    user_timestamp = format_timestamp()
    
    # 显示用户消息在右侧
    with st.chat_message("user"):
        # 实时渲染用户输入的 Markdown 内容
        st.markdown(prompt)
        st.caption(f"{user_timestamp}")

        # 添加用户消息
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "timestamp": user_timestamp}
        )

    # 获取模型回复 - 助手消息显示在左侧
    with st.chat_message("assistant"):
        # 创建占位符用于流式显示
        thinking_placeholder = st.empty()
        response_placeholder = st.empty()
        meta_placeholder = st.empty()
        start_time = time.time()
        
        # 发送请求并处理流式响应
        try:
            # 首先显示spinner
            with st.spinner("Thinking...", show_time=True):
                # 设置为流式请求，并携带最近上下文
                conversation_prompt = build_conversation_prompt(
                    st.session_state.messages, prompt
                )
                data = {
                    "model": model_name,
                    "prompt": conversation_prompt,
                    "stream": True,  # 开启流式响应
                    "options": {
                        "temperature": temperature
                    }
                }
                
                # 发送请求（spinner会在这里结束）
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json=data,
                    stream=True,  # 开启流式请求
                    timeout=120
                )
                response.raise_for_status()
            
            # spinner已关闭，开始处理流式响应
            # 处理流式响应
            full_thinking = ""
            full_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    # 解析每个chunk
                    chunk_data = json.loads(chunk)
                    # print(chunk_data)

                    if "thinking" in chunk_data:
                        # 累加响应内容
                        full_thinking += chunk_data["thinking"]
                        formatted_thinking = format_thinking_markdown(full_thinking)
                        if formatted_thinking:
                            thinking_placeholder.markdown(formatted_thinking)

                    if "response" in chunk_data and "thinking" not in chunk_data:
                        # 累加响应内容
                        full_response += chunk_data["response"]
                        # 更新显示
                        response_placeholder.markdown(full_response)
                    
                    # 如果收到完成标志，退出循环
                    if chunk_data.get("done", False):
                        break
                    # 短暂延迟，使流式效果更明显
                    time.sleep(0.05)
            
            assistant_timestamp = format_timestamp()
            elapsed = time.time() - start_time
            formatted_thinking = format_thinking_markdown(full_thinking)
            content_parts = []
            if formatted_thinking:
                content_parts.append(formatted_thinking)
            if full_response:
                content_parts.append(full_response)
            final_content = "\n\n".join(content_parts).strip()

            # 使用占位符更新元信息，避免重复创建元素
            st.caption(
                f"{assistant_timestamp} spend: {elapsed:.1f} s"
            )

            # 确保响应被保存到会话状态
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_content,
                    "timestamp": assistant_timestamp,
                    "duration": elapsed,
                }
            )
                
        except Exception as e:
            error_msg = f"请求失败: {str(e)}"
            st.error(error_msg)

    
