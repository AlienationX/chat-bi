import threading
import time
from pathlib import Path

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from chainlit.input_widget import Select, Slider, Switch, Tags
from chainlit.types import ThreadDict
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer
from mcp import ClientSession

# 导入兼容OpenAI的客户端库
from openai import AsyncOpenAI

# 全局变量存储 MCP 服务线程
_mcp_threads = []
_mcp_servers = []

"""chainlit run app.py --port 8000 --workers 2"""

# 设置 Chainlit JWT secret（用于密码认证）
# 如果环境变量中已设置，则使用环境变量中的值；否则使用默认值

commands = [
    {"id": "Picture", "icon": "image", "description": "Use DALL-E"},
    {"id": "Search", "icon": "globe", "description": "Find on the web"},
    {
        "id": "Canvas",
        "icon": "pen-line",
        "description": "Collaborate on writing and code",
    },
    {"id": "Database", "icon": "database", "description": "Use database"},
]


# 自定义本地存储客户端
class LocalStorageClient(BaseStorageClient):
    """本地文件系统存储客户端"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload_file(self, file_path: str, file_content: bytes) -> str:
        """上传文件到本地存储"""
        target_path = self.base_path / file_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(file_content)
        return str(target_path)

    async def get_read_url(self, file_path: str) -> str:
        """获取文件的读取URL"""
        return str(self.base_path / file_path)

    async def delete_file(self, file_path: str) -> None:
        """删除文件"""
        target_path = self.base_path / file_path
        if target_path.exists():
            target_path.unlink()

    async def close(self) -> None:
        """关闭存储客户端"""
        pass


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print(">>> tool message: get_weather", city)
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


def _run_mcp_server(module_name: str, server_name: str, port: int, transport: str = "streamable-http"):
    """在后台线程中运行 MCP 服务器"""
    try:
        # 动态导入 MCP 模块
        import importlib

        module_path = f"mcp_servers.{module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as import_error:
            # 如果导入失败，尝试直接导入模块
            print(f"⚠️ 尝试导入 {module_path} 失败: {import_error}")
            # 尝试使用 sys.path 添加当前目录
            import sys
            from pathlib import Path

            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            module = importlib.import_module(f"mcp_servers.{module_name}")

        # 获取 mcp 对象
        mcp_instance = getattr(module, "mcp", None)
        if mcp_instance is None:
            print(f"❌ 未找到 {server_name} MCP 服务器实例 (模块: {module_path})")
            return

        # 直接修改端口设置
        if hasattr(mcp_instance, "settings"):
            mcp_instance.settings.port = port
            mcp_instance.settings.host = "127.0.0.1"
        else:
            print(f"⚠️ {server_name} MCP 服务器实例没有 settings 属性，使用默认端口")

        print(f"🚀 启动 {server_name} MCP 服务器 (transport: {transport}, port: {port})...")
        # 运行 MCP 服务器（阻塞调用）
        mcp_instance.run(transport=transport)
    except Exception as e:
        print(f"❌ {server_name} MCP 服务器启动失败: {str(e)}")
        import traceback

        traceback.print_exc()


def start_mcp():
    """启动 MCP 服务器"""
    global _mcp_threads, _mcp_servers

    print("🚀 正在启动 MCP 服务器...")

    # 定义要启动的 MCP 服务 (模块名, 服务器名, 端口, 传输方式)
    # 从 8001 开始分配端口，避免与 Chainlit 的 8000 端口冲突
    mcp_services = [
        ("db_postgresql", "PostgreSQL Database", 8001, "streamable-http"),
        ("math", "Math", 8002, "streamable-http"),
    ]

    # 为每个服务创建后台线程
    for module_name, server_name, port, transport in mcp_services:
        thread = threading.Thread(
            target=_run_mcp_server,
            args=(module_name, server_name, port, transport),
            daemon=True,  # 设置为守护线程，主进程退出时自动终止
            name=f"MCP-{server_name}",
        )
        thread.start()
        _mcp_threads.append(thread)
        print(f"✅ {server_name} MCP 服务器线程已启动 (端口: {port})")

    # 等待一小段时间确保服务器启动
    time.sleep(1)
    print(f"✅ 已启动 {len(_mcp_threads)} 个 MCP 服务器")


def stop_mcp():
    """停止 MCP 服务器"""
    global _mcp_threads, _mcp_servers

    print("🛑 正在停止 MCP 服务器...")

    # 由于使用守护线程，主进程退出时会自动终止
    # 但我们可以显式地标记它们
    for thread in _mcp_threads:
        if thread.is_alive():
            print(f"🛑 停止 MCP 服务器线程: {thread.name}")

    _mcp_threads.clear()
    _mcp_servers.clear()
    print("✅ MCP 服务器已停止")


@cl.step(type="tool")
async def tool():
    print("tool called...")
    await cl.sleep(1)
    return "Response from the tool!"


# 这个函数只在应用启动时执行一次
@cl.on_app_startup
async def app_startup():
    start_mcp()
    print("✅ 应用启动：AI客户端已初始化")


@cl.on_app_shutdown
async def app_shutdown():
    """应用关闭时执行清理"""
    print("🛑 应用关闭：正在停止 MCP 服务器...")
    stop_mcp()
    print("✅ 应用已关闭")


# 密码认证回调函数
@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    print(f">>> auth_callback user={username} pwd={password}")
    print(">>>", cl.User(identifier=username))

    return cl.User(identifier=username)
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    # import hashlib
    # import json
    # from pathlib import Path

    # from sqlalchemy import text
    # from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    # from sqlalchemy.orm import sessionmaker

    # # 密码哈希函数
    # def hash_password(raw_password):
    #     return hashlib.sha256(raw_password.encode("utf-8")).hexdigest()

    # # 简单的默认用户验证（用于开发环境）
    # # 如果数据库中没有用户，使用默认验证
    # if (username, password) == ("admin", "admin"):
    #     return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})

    # # 尝试从数据库验证用户
    # db_path = Path(__file__).parent / "data" / "chat_sessions.db"
    # if not db_path.exists():
    #     # 数据库不存在，使用默认验证
    #     return None

    # try:
    #     db_url = f"sqlite+aiosqlite:///{db_path}"
    #     engine = create_async_engine(db_url, echo=False)
    #     async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    #     async with async_session() as session:
    #         # 查询用户（users 表只有 identifier, metadata, createdAt 字段）
    #         stmt = text("SELECT identifier, metadata FROM users WHERE identifier=:uname")
    #         result = await session.execute(stmt, {"uname": username})
    #         user_row = result.fetchone()

    #         if user_row is None:
    #             return None

    #         # 用户存在，从 metadata 中获取密码哈希
    #         db_username, db_metadata_raw = user_row

    #         # 解析 metadata（可能是 JSON 字符串或字典）
    #         if isinstance(db_metadata_raw, str):
    #             try:
    #                 db_metadata = json.loads(db_metadata_raw)
    #             except json.JSONDecodeError:
    #                 db_metadata = {}
    #         else:
    #             db_metadata = db_metadata_raw or {}

    #         # 从 metadata 中获取密码哈希
    #         db_password_hash = db_metadata.get("password_hash")

    #         # 如果 metadata 中没有密码哈希，跳过密码验证（仅用于已存在的用户）
    #         if db_password_hash is None:
    #             # 如果没有设置密码，允许登录（首次登录）
    #             return cl.User(identifier=db_username, metadata=db_metadata)

    #         # 验证密码
    #         if db_password_hash == hash_password(password):
    #             return cl.User(identifier=db_username, metadata=db_metadata)
    #         else:
    #             return None
    # except Exception as e:
    #     print(f"数据库验证错误: {e}")
    #     # 如果数据库查询失败，回退到默认验证
    #     return None


# 配置数据持久化
@cl.data_layer
def get_data_layer():
    # 配置 SQLite 数据库连接
    data_path = Path(__file__).parent / "data"
    data_path.mkdir(exist_ok=True)  # 确保数据目录存在
    sqlite_conninfo = f"sqlite+aiosqlite:///./{data_path.name}/chat_sessions.db"
    sqlite_conninfo_with_params = f"{sqlite_conninfo}?cache=shared&journal_mode=WAL"

    # 配置本地文件存储路径
    local_storage_path = f"./{data_path.name}/uploads"  # 存储上传文件的目录
    # 创建本地存储客户端
    storage_client = LocalStorageClient(local_storage_path)
    # 创建 SQLite 数据层
    data_layer = SQLAlchemyDataLayer(conninfo=sqlite_conninfo_with_params, storage_provider=storage_client)

    return data_layer


# @cl.set_chat_profiles
# async def chat_profile(current_user: cl.User):
#     # if current_user.metadata["role"] != "ADMIN":
#     #     return None

#     return [
#         cl.ChatProfile(
#             name="My Chat Profile",
#             icon="https://picsum.photos/250",
#             markdown_description="The underlying LLM model is **GPT-3.5**, a *175B parameter model* trained on 410GB of text data.",
#             starters=[
#                 cl.Starter(
#                     label="Morning routine ideation",
#                     message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
#                     icon="/public/idea.svg",
#                 ),
#                 cl.Starter(
#                     label="Explain superconductors",
#                     message="Explain superconductors like I'm five years old.",
#                     icon="/public/learn.svg",
#                 ),
#             ],
#         ),
#         cl.ChatProfile(
#             name="GPT-5",
#             markdown_description="The underlying LLM model is **GPT-5**.",
#             icon="https://picsum.photos/200",
#         ),
#     ]


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/terminal.svg",
            command="code",
        ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/write.svg",
        ),
    ]


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    print(">>> on_mcp_connect", connection.name)
    # List available tools
    result = await session.list_tools()

    # Process tool metadata
    tools = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        }
        for t in result.tools
    ]

    # Store tools for later use
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    print(">>> mcp_tools", mcp_tools)
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Your cleanup code here
    # This handler is optional
    print(">>> on_mcp_disconnect", name)


@cl.on_chat_start
async def chat_start():
    print(">>> on_chat_start, 新建会话也会触发！！！")

    # 设置命令
    await cl.context.emitter.set_commands(commands)

    cl.user_session.set("chat_history", [])

    settings = cl.user_session.get("settings", {})
    if not settings:
        settings = {
            "Model": "deepseek-r1:8b",
            "Think": True,
            "Temperature": 0.7,
            "Tags": ["Answer", "Chat-BI"],
        }
        cl.user_session.set("settings", settings)

    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Select a model",
                values=[
                    "deepseek-r1:8b",
                    "deepseek-v3.1:671b-cloud",
                    "qwen2.5:1.5b",
                ],
                initial_value=settings["Model"],
            ),
            Switch(id="Think", label="Enable thinking for models", initial=settings["Think"]),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=settings["Temperature"],
                min=0,
                max=1,
                step=0.1,
            ),
            Tags(id="Tags", label="OpenAI - StopSequence", initial=settings["Tags"]),
        ]
    ).send()

    # 初始化客户端，关键是指定 base_url 为 Ollama 服务的地址
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",  # Ollama 的 OpenAI 兼容 API 端点
        api_key="unnecessary",  # 本地运行无需有效的 API 密钥，可任意填写
    )
    # 将客户端存入用户会话，方便后续调用

    # ############################################### Create the TaskList
    task_list = cl.TaskList()
    task_list.status = "Running..."

    # Create a task and put it in the running state
    task1 = cl.Task(title="Processing data", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task1)
    # Create another task that is in the ready state
    task2 = cl.Task(title="Performing calculations")
    await task_list.add_task(task2)

    # Optional: link a message to each task to allow task navigation in the chat history
    message = await cl.Message(content="Started processing data").send()
    task1.forId = message.id

    # Update the task list in the interface
    await task_list.send()

    # Perform some action on your end
    await cl.sleep(5)

    # Update the task statuses
    task1.status = cl.TaskStatus.DONE
    task2.status = cl.TaskStatus.FAILED
    task_list.status = "Failed"
    await task_list.send()
    # ###############################################
    await cl.ElementSidebar.set_title("任务面板")
    await cl.ElementSidebar.set_elements(
        [],
        key="task-panel",
    )

    mcp_tools = cl.user_session.get("mcp_tools", [])
    print(">>> chat_start's mcp_tools", mcp_tools)

    llm = ChatOllama(
        # qwen2.5:1.5b, deepseek-v3.1:671b-cloud, deepseek-r1:8b
        model="deepseek-r1:8b",  # 选择任何你本地已有的模型,deepseek-r1:8b不支持工具调用
        temperature=0.7,  # 控制生成结果的随机性
        num_ctx=4096,  # 设置上下文窗口大小
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather] + mcp_tools,
        system_prompt="You are a helpful assistant",
    )
    cl.user_session.set("agent", agent)

    # await cl.Message(content="你好！我已连接本地模型，请开始提问。").send()


@cl.on_chat_resume
async def chat_resume(thread: ThreadDict):
    print(">>> on_chat_resume")

    # 恢复设置
    settings = cl.user_session.get("settings")
    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Select a model",
                values=[
                    "deepseek-r1:8b",
                    "deepseek-v3.1:671b-cloud",
                    "qwen2.5:1.5b",
                ],
                initial_value=settings["Model"],
            ),
            Switch(id="Think", label="Enable thinking for models", initial=settings["Think"]),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=settings["Temperature"],
                min=0,
                max=1,
                step=0.1,
            ),
            Tags(id="Tags", label="OpenAI - StopSequence", initial=settings["Tags"]),
        ]
    ).send()

    # 恢复聊天历史
    for message in thread.get("steps", []):
        if message.get("type") == "user_message":
            cl.user_session.get("chat_history").append(
                {
                    "role": "user",
                    "content": message.get("content"),
                }
            )
        elif message.get("type") == "assistant_message":
            cl.user_session.get("chat_history").append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                }
            )
        else:
            cl.user_session.get("chat_history").append(
                {
                    "role": "system",
                    "content": message.get("content"),
                }
            )


@cl.on_settings_update
async def settings_update(settings):
    # 更新设置
    print(">>> on_settings_update", settings)
    cl.user_session.set("settings", settings)
    # TODO: 更新和存储到 cl.User.metadata 中
    # cl.user_session.get("user").metadata = settings


@cl.on_message
async def main(message: cl.Message):
    try:
        # 处理命令/指令
        if message.command == "Database":
            # User is using the Picture command
            print(">>> Database command used")

        chat_history = cl.user_session.get("chat_history")
        chat_history.append(
            {
                "role": "user",
                "content": message.content,
            }
        )

        settings = cl.user_session.get("settings")
        model = settings["Model"]
        print(">>>> settings", settings)
        print(">>>> model", model)

        # qwen2.5:1.5b 不是推理模型，不支持 thinking
        settings["Think"] = False if settings["Think"] and model == "qwen2.5:1.5b" else settings["Think"]

        # ###########询问用户是否继续#################
        # res = await cl.AskActionMessage(
        #     content="Pick an action!",
        #     actions=[
        #         cl.Action(name="continue", payload={"value": "continue"}, label="✅ Continue"),
        #         cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
        #     ],
        # ).send()

        # if res and res.get("payload").get("value") == "continue":
        #     await cl.Message(
        #         content="Continue!",
        #     ).send()
        # ###########################################

        llm = ChatOllama(
            # qwen2.5:1.5b, deepseek-v3.1:671b-cloud, deepseek-r1:8b
            model=model,  # 选择任何你本地已有的模型,deepseek-r1:8b不支持工具调用
            temperature=settings["Temperature"],  # 控制生成结果的随机性
            reasoning=settings["Think"],
            # num_ctx=4096,  # 设置上下文窗口大小
        )

        agent = create_agent(
            model=llm,
            tools=[get_weather] if model != "deepseek-r1:8b" else [],
            system_prompt="You are a helpful assistant",
        )

        stream = agent.stream(
            {"messages": chat_history},
            stream_mode="messages",
        )

        assistant_response = ""
        msg = cl.Message(content="")

        if settings["Think"]:
            start_time = time.time()
            async with cl.Step(name="Thinking", type="llm") as thinking_step:
                for token, metadata in stream:
                    reasoning = token.content_blocks[-1].get("reasoning", "") if token.content_blocks else ""
                    if reasoning:
                        # 流式传输思考内容
                        await thinking_step.stream_token(reasoning)
                        # thinking_step.output = reasoning

                    content = token.content_blocks[-1].get("text", "") if token.content_blocks else ""
                    if content:
                        # 更新思考步骤名称和时间
                        thought_duration = round(time.time() - start_time)
                        thinking_step.name = f"Thought for {thought_duration}s"
                        await thinking_step.update()

                        # 更新首个content
                        assistant_response += content
                        await msg.stream_token(content)
                        # 关闭thinking_step
                        break

        # 流式传输 assistant_response 内容
        for token, metadata in stream:
            content = token.content_blocks[-1].get("text", "") if token.content_blocks else ""
            if content:
                assistant_response += content
                await msg.stream_token(content)

        chat_history.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )
        await msg.update()

        # # 判断是否调用工具
        # # if chunk.choices[0].delta.tool_calls is not None:
        # tool_res = await tool()
        # await cl.Message(content=tool_res).send()

    except Exception:
        import traceback

        traceback.print_exc()
        await cl.Message(content=f"出现异常: {traceback.format_exc()}").send()
        # await cl.Message(content=f"出现异常: {e}").send()
