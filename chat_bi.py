import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import pymysql
import streamlit as st
from langchain.agents import create_agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from sqlalchemy import create_engine

# 页面配置
st.set_page_config(page_title="Chat BI", page_icon="📊", layout="wide")

# 自定义样式
st.markdown(
    """
    <style>
        .stAppDeployButton, .stStatusWidget {
            display: none !important;
        }
        [data-testid="stChatMessageContent"] {
            font-size: 0.90em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_timestamp(ts: Optional[datetime] = None) -> str:
    """格式化时间戳"""
    return (ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def get_sql_database_uri() -> str:
    """生成 SQLDatabase URI"""
    return "mysql+pymysql://readonly:123456@192.168.9.88:3306/admin_vip?charset=utf8mb4"


def get_db_engine():
    """连接 MySQL 数据库"""
    db_engine = create_engine(get_sql_database_uri())
    return db_engine


def detect_query_intent(user_input: str, llm: Ollama) -> Dict[str, Any]:
    """检测用户意图：是查询数据还是生成图表"""
    prompt_template = """
分析用户的输入，判断用户的意图。用户输入：{user_input}

请以 JSON 格式返回，包含以下字段：
- intent: "query" (查询数据) 或 "chart" (生成图表) 或 "chat" (普通对话)
- chart_type: 如果是图表，指定类型 "line", "bar", "area", "pie", "scatter" 等，否则为 null
- needs_sql: true 或 false，是否需要执行 SQL 查询

只返回 JSON，不要其他文字。
示例：{{"intent": "chart", "chart_type": "line", "needs_sql": true}}
"""

    try:
        # 直接使用 LLM 的 invoke 方法，不需要 LLMChain
        formatted_prompt = prompt_template.format(user_input=user_input)
        result = llm.invoke(formatted_prompt)
        # 提取 JSON
        json_match = re.search(r"\{.*\}", result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"intent": "chat", "chart_type": None, "needs_sql": False}
    except Exception as e:
        st.warning(f"意图检测失败: {str(e)}")
        return {"intent": "chat", "chart_type": None, "needs_sql": False}


def execute_sql_query(query: str, connection: pymysql.Connection) -> Optional[pd.DataFrame]:
    """执行 SQL 查询并返回 DataFrame"""
    try:
        df = pd.read_sql(query, connection)
        print(query, df)
        return df
    except pymysql.Error as e:
        st.error(f"SQL 执行失败: {str(e)}")
        return None
    except Exception as e:
        st.error(f"查询失败: {str(e)}")
        return None


def generate_sql_from_natural_language(user_input: str, db: SQLDatabase, llm: Ollama) -> Optional[str]:
    """使用 LangChain 从自然语言生成 SQL"""
    try:
        # 创建 SQL Agent
        agent = create_sql_agent(
            llm=llm,
            db=db,
            verbose=False,
            agent_type="openai-tools",
        )
        print(agent)

        # 生成 SQL
        result = agent.invoke({"input": user_input})
        print(result)

        # 从结果中提取 SQL
        if isinstance(result, dict):
            # 尝试从 intermediate_steps 中提取 SQL
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if isinstance(step, tuple) and len(step) > 0:
                        action = step[0]
                        if hasattr(action, "tool_input"):
                            sql = action.tool_input
                            if isinstance(sql, dict) and "query" in sql:
                                return sql["query"]
                            elif isinstance(sql, str) and sql.strip().upper().startswith("SELECT"):
                                return sql

            # 从输出中提取 SQL
            output = result.get("output", "")
            sql_match = re.search(r"SELECT.*?;", output, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group()

            # 如果没有分号，尝试提取到换行或结束
            sql_match = re.search(r"SELECT.*", output, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group().strip()

        # 如果 result 是字符串
        if isinstance(result, str):
            sql_match = re.search(r"SELECT.*?;", result, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group()

        return None
    except Exception as e:
        st.error(f"SQL 生成失败: {str(e)}")
        return None


def display_chart(df: pd.DataFrame, chart_type: str, x_column: Optional[str] = None, y_columns: Optional[list] = None):
    """根据图表类型显示图表"""
    if df.empty:
        st.warning("数据为空，无法生成图表")
        return

    # 自动选择列
    if x_column is None:
        # 尝试找到日期/时间列作为 x 轴
        date_cols = df.select_dtypes(include=["datetime64", "object"]).columns
        if len(date_cols) > 0:
            x_column = date_cols[0]
        else:
            x_column = df.columns[0]

    if y_columns is None:
        # 选择数值列作为 y 轴
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols) > 0:
            y_columns = numeric_cols[:3]  # 最多3个系列
        else:
            st.warning("未找到数值列用于图表")
            return

    try:
        chart_df = df[[x_column] + y_columns].copy()
        chart_df = chart_df.set_index(x_column)

        if chart_type == "line":
            st.line_chart(chart_df)
        elif chart_type == "bar":
            st.bar_chart(chart_df)
        elif chart_type == "area":
            st.area_chart(chart_df)
        elif chart_type == "scatter":
            if len(y_columns) >= 2:
                st.scatter_chart(chart_df[[y_columns[0], y_columns[1]]])
            else:
                st.warning("散点图需要至少2个数值列")
        elif chart_type == "pie":
            if len(y_columns) > 0:
                # 饼图使用第一个数值列
                try:
                    import plotly.express as px

                    pie_df = pd.DataFrame({"value": chart_df[y_columns[0]].values, "label": chart_df.index})
                    fig = px.pie(pie_df, values="value", names="label", title=f"{y_columns[0]} 分布")
                    st.plotly_chart(fig)
                except ImportError:
                    st.warning("饼图需要 plotly 库，请安装: uv add plotly")
            else:
                st.warning("饼图需要数值列")
        else:
            st.line_chart(chart_df)  # 默认折线图
    except Exception as e:
        st.error(f"图表生成失败: {str(e)}")
        st.dataframe(df)


# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_engine" not in st.session_state:
    st.session_state.db_engine = get_db_engine()

if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

# 侧边栏配置
with st.sidebar:
    # Ollama 配置
    # 获取Ollama模型列表
    def get_ollama_models(ollama_url):
        try:
            import requests

            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                if "models" in models_data:
                    return [model["name"] for model in models_data["models"]]
        except Exception as e:
            st.warning(f"无法获取模型列表: {str(e)}")
        return []

    st.title("🤖 模型配置")
    ollama_model = st.selectbox(
        "选择 Ollama 模型",
        options=get_ollama_models("http://localhost:11434"),
        index=0,
    )
    ollama_base_url = st.text_input("Ollama API 地址", value="http://localhost:11434")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

# 主界面
st.title("📊 Chat BI - 智能数据分析")
st.caption("使用自然语言查询数据库并生成图表")

# 显示聊天记录
for message in st.session_state.messages:
    role = message["role"]
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")

    with st.chat_message(role):
        st.markdown(content)
        if timestamp:
            st.caption(timestamp)

        # 如果是助手消息且包含数据，显示 DataFrame 或图表
        if role == "assistant":
            if "dataframe" in message:
                st.dataframe(message["dataframe"])
            if "chart_type" in message and message["chart_type"]:
                display_chart(
                    message.get("dataframe", pd.DataFrame()),
                    message["chart_type"],
                    message.get("x_column"),
                    message.get("y_columns"),
                )

# 聊天输入
if prompt := st.chat_input("请输入你的问题或查询..."):
    user_timestamp = format_timestamp()

    # 添加用户消息
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "timestamp": user_timestamp,
        }
    )

    # 显示用户消息
    with st.chat_message("user"):
        # st.markdown(prompt)
        st.success(prompt)
        st.caption(user_timestamp)

    # 检查数据库是否已连接
    if not st.session_state.db_initialized:
        with st.chat_message("assistant"):
            st.error("请先在侧边栏连接数据库")
        st.stop()

    # 初始化 LLM
    try:
        llm = Ollama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature,
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"LLM 初始化失败: {str(e)}")
        st.stop()

    # 处理助手回复
    with st.chat_message("assistant"):
        assistant_timestamp = format_timestamp()

        # 检测用户意图
        intent_result = detect_query_intent(prompt, llm)
        intent = intent_result.get("intent", "chat")
        chart_type = intent_result.get("chart_type")
        needs_sql = intent_result.get("needs_sql", False)

        if intent == "query" or intent == "chart":
            # 需要查询数据库
            try:
                # 创建 SQLDatabase
                db_uri = get_sql_database_uri()
                db = SQLDatabase.from_uri(db_uri)

                # 生成 SQL
                with st.spinner("正在生成 SQL 查询..."):
                    sql_query = generate_sql_from_natural_language(prompt, db, llm)
                    # sql_query = "SELECT * FROM tj_member_order_info limit 10;"

                if sql_query:
                    st.code(sql_query, language="sql")

                    # 执行 SQL
                    with st.spinner("正在执行查询..."):
                        df = execute_sql_query(sql_query, st.session_state.db_engine)

                    if df is not None and not df.empty:
                        # 显示数据
                        st.dataframe(df)

                        # 如果是图表意图，生成图表
                        if intent == "chart" and chart_type:
                            display_chart(df, chart_type)

                        # 保存到消息历史
                        response_content = f"查询完成，共 {len(df)} 行数据。"
                        if intent == "chart":
                            response_content += f"\n\n已生成 {chart_type} 图表。"

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_content,
                                "timestamp": assistant_timestamp,
                                "dataframe": df,
                                "chart_type": chart_type if intent == "chart" else None,
                            }
                        )
                    else:
                        error_msg = "查询结果为空或查询失败。"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_msg,
                                "timestamp": assistant_timestamp,
                            }
                        )
                else:
                    error_msg = "无法生成有效的 SQL 查询，请尝试更明确的描述。"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": assistant_timestamp,
                        }
                    )
            except Exception as e:
                error_msg = f"处理查询时出错: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": assistant_timestamp,
                    }
                )
        else:
            # 普通对话
            try:
                with st.spinner("正在思考..."):
                    response = llm.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "timestamp": assistant_timestamp,
                        }
                    )
            except Exception as e:
                error_msg = f"生成回复时出错: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": assistant_timestamp,
                    }
                )

        st.caption(assistant_timestamp)

# 侧边栏操作
with st.sidebar:
    st.divider()
    if st.button("清空对话记录"):
        st.session_state.messages = []
        st.rerun()
