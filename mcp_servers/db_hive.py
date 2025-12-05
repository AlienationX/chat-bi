import json
import os
from typing import Optional

try:
    from pyhive import hive
    from TCLIService.ttypes import TOperationState
except ImportError:
    # 如果 pyhive 未安装，提供友好的错误提示
    hive = None
    TOperationState = None

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Hive Database")


def get_db_connection(database: Optional[str] = None):
    """获取 Hive 数据库连接"""
    if hive is None:
        raise Exception("pyhive 库未安装，请运行: pip install pyhive")
    
    try:
        conn = hive.connect(
            host=os.getenv("HIVE_HOST", "localhost"),
            port=int(os.getenv("HIVE_PORT", "10000")),
            database=database or os.getenv("HIVE_DB", "default"),
            username=os.getenv("HIVE_USER", ""),
            password=os.getenv("HIVE_PASSWORD", ""),
            auth=os.getenv("HIVE_AUTH", "NOSASL"),  # NOSASL, LDAP, KERBEROS, etc.
        )
        return conn
    except Exception as e:
        raise Exception(f"数据库连接失败: {str(e)}")


def wait_for_operation(cursor):
    """等待 Hive 操作完成"""
    status = cursor.poll().operationState
    while status in (TOperationState.INITIALIZED_STATE, TOperationState.RUNNING_STATE):
        status = cursor.poll().operationState
    return status == TOperationState.FINISHED_STATE


@mcp.tool()
async def execute_query(query: str, limit: int = 1000) -> str:
    """
    执行 SQL 查询并返回结果。

    Args:
        query: 要执行的 SQL 查询语句（HiveQL）
        limit: 限制返回的行数（默认 1000）

    Returns:
        JSON 格式的查询结果
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 对于 SELECT 查询，添加 LIMIT
        query_upper = query.strip().upper()
        if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
            # 简单检查，避免在子查询中添加 LIMIT
            if query_upper.count("SELECT") == 1:
                query = f"{query.rstrip(';')} LIMIT {limit}"

        cursor.execute(query, async_=True)
        
        # 等待查询完成
        if not wait_for_operation(cursor):
            raise Exception("查询执行失败")

        # 如果是 SELECT 查询，获取结果
        if query_upper.startswith("SELECT"):
            # 获取列名
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            # 将结果转换为字典列表
            data = []
            for row in rows:
                data.append(dict(zip(columns, row)))
            
            result = {
                "success": True,
                "row_count": len(data),
                "data": data,
            }
        else:
            # 对于 INSERT, CREATE, DROP 等操作
            result = {
                "success": True,
                "message": "操作执行成功",
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@mcp.tool()
async def list_databases() -> str:
    """
    列出所有可用的数据库。

    Returns:
        JSON 格式的数据库列表
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES", async_=True)
        
        if not wait_for_operation(cursor):
            raise Exception("查询执行失败")
        
        databases = [row[0] for row in cursor.fetchall()]

        result = {
            "success": True,
            "databases": databases,
            "count": len(databases),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@mcp.tool()
async def list_tables(database: Optional[str] = None) -> str:
    """
    列出指定数据库中的所有表。

    Args:
        database: 数据库名称（可选，默认使用配置的数据库）

    Returns:
        JSON 格式的表列表
    """
    conn = None
    cursor = None
    try:
        db_name = database or os.getenv("HIVE_DB", "default")
        conn = get_db_connection(db_name)
        cursor = conn.cursor()
        
        # 切换到指定数据库
        if db_name != "default":
            cursor.execute(f"USE {db_name}", async_=True)
            wait_for_operation(cursor)
        
        cursor.execute("SHOW TABLES", async_=True)
        
        if not wait_for_operation(cursor):
            raise Exception("查询执行失败")
        
        tables = [{"name": row[0], "type": "TABLE"} for row in cursor.fetchall()]

        result = {
            "success": True,
            "database": db_name,
            "tables": tables,
            "count": len(tables),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@mcp.tool()
async def describe_table(table: str, database: Optional[str] = None) -> str:
    """
    获取表的详细结构信息，包括列名、数据类型等。

    Args:
        table: 表名
        database: 数据库名称（可选，默认使用配置的数据库）

    Returns:
        JSON 格式的表结构信息
    """
    conn = None
    cursor = None
    try:
        db_name = database or os.getenv("HIVE_DB", "default")
        conn = get_db_connection(db_name)
        cursor = conn.cursor()
        
        # 切换到指定数据库
        if db_name != "default":
            cursor.execute(f"USE {db_name}", async_=True)
            wait_for_operation(cursor)
        
        # 获取表结构
        cursor.execute(f"DESCRIBE {table}", async_=True)
        
        if not wait_for_operation(cursor):
            raise Exception("查询执行失败")
        
        columns = []
        for row in cursor.fetchall():
            if len(row) >= 2:
                columns.append({
                    "column_name": row[0],
                    "data_type": row[1],
                    "comment": row[2] if len(row) > 2 else None,
                })
        
        # 获取分区信息
        partitions = []
        try:
            cursor.execute(f"SHOW PARTITIONS {table}", async_=True)
            if wait_for_operation(cursor):
                partitions = [row[0] for row in cursor.fetchall()]
        except Exception:
            # 表可能没有分区
            pass
        
        # 获取表格式信息
        table_format = None
        try:
            cursor.execute(f"SHOW CREATE TABLE {table}", async_=True)
            if wait_for_operation(cursor):
                create_table_sql = "\n".join([row[0] for row in cursor.fetchall()])
                table_format = create_table_sql
        except Exception:
            pass

        result = {
            "success": True,
            "table": table,
            "database": db_name,
            "columns": columns,
            "partitions": partitions,
            "partition_count": len(partitions),
            "create_table_sql": table_format,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@mcp.tool()
async def get_table_sample(table: str, database: Optional[str] = None, limit: int = 10) -> str:
    """
    获取表的示例数据（前 N 行）。

    Args:
        table: 表名
        database: 数据库名称（可选，默认使用配置的数据库）
        limit: 返回的行数（默认 10）

    Returns:
        JSON 格式的示例数据
    """
    conn = None
    cursor = None
    try:
        db_name = database or os.getenv("HIVE_DB", "default")
        conn = get_db_connection(db_name)
        cursor = conn.cursor()
        
        # 切换到指定数据库
        if db_name != "default":
            cursor.execute(f"USE {db_name}", async_=True)
            wait_for_operation(cursor)
        
        query = f"SELECT * FROM {table} LIMIT {limit}"
        cursor.execute(query, async_=True)
        
        if not wait_for_operation(cursor):
            raise Exception("查询执行失败")
        
        # 获取列名
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        
        # 将结果转换为字典列表
        data = []
        for row in rows:
            data.append(dict(zip(columns, row)))

        result = {
            "success": True,
            "table": table,
            "database": db_name,
            "row_count": len(data),
            "data": data,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
