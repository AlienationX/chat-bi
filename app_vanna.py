import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import os
import shutil
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# 1. 初始化示例数据库
def create_sample_database(db_name='example.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 创建部门信息表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS departments (
        dept_id INTEGER PRIMARY KEY,
        dept_name TEXT NOT NULL,
        manager_name TEXT NOT NULL,
        location TEXT NOT NULL
    )
    ''')
    
    # 创建营业额表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS revenue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dept_id INTEGER,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        revenue_amount DECIMAL(12,2) NOT NULL,
        FOREIGN KEY (dept_id) REFERENCES departments (dept_id)
    )
    ''')
    
    # 插入示例数据
    departments_data = [(1, '销售部', '张经理', '北京'),
                         (2, '市场部', '李经理', '上海'),
                         (3, '技术部', '王经理', '深圳')]
    cursor.executemany('INSERT INTO departments VALUES (?,?,?,?)', departments_data)
    
    revenue_data = []
    for dept_id in [1, 2, 3]:
        # 为每个部门生成2023和2024年的模拟月度营业额数据
        base_revenue_2023 = {1: 150000, 2: 120000, 3: 180000}[dept_id]
        for year in [2023, 2024]:
            growth_rate = {1: -0.05, 2: 0.08, 3: 0.12}[dept_id]  # 各部门增长率不同
            base_revenue = base_revenue_2023 * (1 + growth_rate) if year == 2024 else base_revenue_2023
            for month in range(1, 13):
                # 加入一些随机波动
                seasonal_factor = 1 + 0.1 * ((month % 4) - 2) / 2
                random_factor = 1 + random.uniform(-0.15, 0.15)
                amount = base_revenue * seasonal_factor * random_factor
                revenue_data.append((dept_id, year, month, int(amount)))
    
    cursor.executemany('INSERT INTO revenue (dept_id, year, month, revenue_amount) VALUES (?,?,?,?)', revenue_data)
    conn.commit()
    conn.close()
    print("示例数据库创建完成！")

# 清理旧数据（如果存在），确保每次运行环境干净
db_name = 'vanna_example.db'
if os.path.exists(db_name):
    os.remove(db_name)
create_sample_database(db_name)

# 2. 定义自定义Vanna类，继承向量数据库和LLM能力
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        # 配置默认值
        if config is None:
            config = {}
        # 设置默认模型（如果未指定）
        if 'model' not in config:
            config['model'] = 'qwen2.5:1.5b'
        
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

# 3. 初始化Vanna实例
# 请确保Ollama服务已在运行（例如在终端中执行 `ollama serve`）
# 并使用 `ollama pull qwen2.5:1.5b` 或其他支持的模型
try:
    vn = MyVanna(config={'model': 'qwen2.5:1.5b'})  # 指定使用的本地模型
    vn.connect_to_sqlite(db_name)  # 连接到SQLite数据库
except ConnectionError as e:
    print("\n错误: 无法连接到Ollama服务。请执行以下步骤：")
    print("1. 确保Ollama已安装: https://ollama.com/download")
    print("2. 在单独的终端窗口中启动Ollama服务: `ollama serve`")
    print("3. 拉取所需模型: `ollama pull qwen2.5:1.5b`")
    print("\n如果您已经安装了其他兼容模型，可以修改代码中的model参数。")
    print(f"\n原始错误信息: {e}")
    import sys
    sys.exit(1)

print("成功连接到Ollama服务！")

# 4. 训练Vanna模型：这是提升准确性的关键步骤
# 4.1 导入数据库结构（DDL）
df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL")
for ddl in df_ddl['sql']:
    vn.train(ddl=ddl)  # 训练DDL，让Vanna理解表结构

# 4.2 提供业务文档说明
vn.train(documentation="部门信息表 departments，包含字段：dept_id（部门ID）, dept_name（部门名称）, manager_name（经理姓名）, location（地点）")
vn.train(documentation="营业额表 revenue，包含字段：id, dept_id（部门ID）, year（年份）, month（月份）, revenue_amount（营业额金额）")
vn.train(documentation="数据包含2023年和2024年每个部门每个月的模拟营业额数据")

# 4.3 提供高质量的SQL示例（问题-SQL对）
vn.train(question="查询2024年各部门的总营业额",
         sql="SELECT d.dept_name, SUM(r.revenue_amount) as total_revenue FROM departments d JOIN revenue r ON d.dept_id = r.dept_id WHERE r.year = 2024 GROUP BY d.dept_name")

vn.train(question="比较各部门2023年和2024年的营业额",
         sql="""
         SELECT d.dept_name,
                SUM(CASE WHEN r.year=2023 THEN r.revenue_amount ELSE 0 END) as revenue_2023,
                SUM(CASE WHEN r.year=2024 THEN r.revenue_amount ELSE 0 END) as revenue_2024
         FROM departments d JOIN revenue r ON d.dept_id = r.dept_id
         WHERE r.year IN (2023, 2024)
         GROUP BY d.dept_name
         """)

print("Vanna 模型训练完成！")

# 5. 进行自然语言查询
# 问题1：简单的数据查询
question_1 = "2024年销售部的总营业额是多少？"
print(f"\n问题: {question_1}")
# 修复返回值解包问题 - 捕获所有返回值
try:
    # 尝试获取所有可能的返回值
    response = vn.ask(question_1)
    if isinstance(response, tuple):
        # 根据返回值数量进行适当的解包
        if len(response) >= 3:
            sql_query, result_df, _ = response  # 忽略第三个返回值
        elif len(response) == 2:
            sql_query, result_df = response
        else:
            sql_query = response[0]
            result_df = None
    else:
        sql_query = str(response)
        result_df = None
        
    print(f"生成的SQL: {sql_query}")
    print(f"查询结果:\n{result_df}")
except Exception as e:
    print(f"执行查询时出错: {e}")

# 问题2：复杂的多表查询和可视化请求
question_2 = "比较各部门2023年和2024年的总营业额，用柱状图显示"
print(f"\n问题: {question_2}")
try:
    # 为可视化请求添加错误处理
    response = vn.ask(question_2, visualize=True)
    if isinstance(response, tuple):
        if len(response) >= 3:
            sql_query, result_df, plotly_code = response
        elif len(response) == 2:
            sql_query, result_df = response
            plotly_code = None
        else:
            sql_query = response[0]
            result_df = None
            plotly_code = None
    else:
        sql_query = str(response)
        result_df = None
        plotly_code = None
    
    print(f"生成的SQL: {sql_query}")
    print(f"查询结果:\n{result_df}")
except Exception as e:
    print(f"执行可视化查询时出错: {e}")
    # 设置默认值以避免后续代码出错
    plotly_code = None
    result_df = None

# 如果生成了图表代码，可以保存为HTML文件查看
if plotly_code:
    try:
        fig = vn.get_plotly_figure(plotly_code, result_df)
        fig.write_html("department_revenue_comparison.html")
        print("图表已保存为 'department_revenue_comparison.html'，请在浏览器中打开查看。")
    except Exception as e:
        print(f"保存图表时出错: {e}")

print("\n演示完成！")

from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)
app.run()