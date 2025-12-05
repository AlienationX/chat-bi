import json
import os
from typing import Optional

import pymysql
from mcp.server.fastmcp import FastMCP
from pymysql.cursors import DictCursor

mcp = FastMCP("MySQL Database")


def get_db_connection(database: Optional[str] = None):
    """获取 MySQL 数据库连接"""
    try:
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=database or os.getenv("MYSQL_DB", "mysql"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            charset="utf8mb4",
            cursorclass=DictCursor,
        )
        return conn
    except Exception as e:
        raise Exception(f"数据库连接失败: {str(e)}")


@mcp.tool()
async def execute_query(query: str, limit: int = 1000) -> str:
    """
    执行 SQL 查询并返回结果。

    Args:
        query: 要执行的 SQL 查询语句
        limit: 限制返回的行数（默认 1000）

    Returns:
        JSON 格式的查询结果
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 对于 SELECT 查询，添加 LIMIT
            query_upper = query.strip().upper()
            if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
                # 简单检查，避免在子查询中添加 LIMIT
                if query_upper.count("SELECT") == 1:
                    query = f"{query.rstrip(';')} LIMIT {limit}"

            cursor.execute(query)

            # 如果是 SELECT 查询，获取结果
            if query_upper.startswith("SELECT"):
                rows = cursor.fetchall()
                result = {
                    "success": True,
                    "row_count": len(rows),
                    "data": rows,
                }
            else:
                # 对于 INSERT, UPDATE, DELETE 等操作
                conn.commit()
                result = {
                    "success": True,
                    "rows_affected": cursor.rowcount,
                    "message": "操作执行成功",
                }

            return json.dumps(result, ensure_ascii=False, indent=2)

    except pymysql.Error as e:
        if conn:
            conn.rollback()
        error_result = {
            "success": False,
            "error": str(e),
            "error_code": e.args[0] if e.args else None,
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    except Exception as e:
        if conn:
            conn.rollback()
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
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
    try:
        # 连接到默认数据库来查询数据库列表
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database="information_schema",  # 连接到系统数据库
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            charset="utf8mb4",
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT SCHEMA_NAME as database_name FROM information_schema.SCHEMATA WHERE SCHEMA_NAME NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys') ORDER BY SCHEMA_NAME"
            )
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
    try:
        db_name = database or os.getenv("MYSQL_DB", "mysql")
        conn = get_db_connection(db_name)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT TABLE_NAME as table_name, TABLE_TYPE as table_type
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = %s
                ORDER BY TABLE_NAME
                """,
                (db_name,),
            )
            tables = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]

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
        if conn:
            conn.close()


@mcp.tool()
async def describe_table(table: str, database: Optional[str] = None) -> str:
    """
    获取表的详细结构信息，包括列名、数据类型、约束等。

    Args:
        table: 表名
        database: 数据库名称（可选，默认使用配置的数据库）

    Returns:
        JSON 格式的表结构信息
    """
    conn = None
    try:
        db_name = database or os.getenv("MYSQL_DB", "mysql")
        conn = get_db_connection(db_name)
        with conn.cursor() as cursor:
            # 获取列信息
            cursor.execute(
                """
                SELECT 
                    COLUMN_NAME as column_name,
                    DATA_TYPE as data_type,
                    CHARACTER_MAXIMUM_LENGTH as character_maximum_length,
                    IS_NULLABLE as is_nullable,
                    COLUMN_DEFAULT as column_default,
                    COLUMN_TYPE as column_type,
                    COLUMN_KEY as column_key,
                    EXTRA as extra
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION
                """,
                (db_name, table),
            )
            columns = [dict(row) for row in cursor.fetchall()]

            # 获取主键信息
            cursor.execute(
                """
                SELECT COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = %s 
                    AND TABLE_NAME = %s
                    AND CONSTRAINT_NAME = 'PRIMARY'
                ORDER BY ORDINAL_POSITION
                """,
                (db_name, table),
            )
            primary_keys = [row[0] for row in cursor.fetchall()]

            # 获取外键信息
            cursor.execute(
                """
                SELECT
                    COLUMN_NAME as column_name,
                    REFERENCED_TABLE_SCHEMA as foreign_table_schema,
                    REFERENCED_TABLE_NAME as foreign_table_name,
                    REFERENCED_COLUMN_NAME as foreign_column_name
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = %s
                    AND TABLE_NAME = %s
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """,
                (db_name, table),
            )
            foreign_keys = [dict(row) for row in cursor.fetchall()]

            # 获取索引信息
            cursor.execute(
                """
                SELECT
                    INDEX_NAME as index_name,
                    COLUMN_NAME as column_name,
                    NON_UNIQUE as non_unique,
                    SEQ_IN_INDEX as seq_in_index
                FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY INDEX_NAME, SEQ_IN_INDEX
                """,
                (db_name, table),
            )
            index_rows = cursor.fetchall()
            # 按索引名分组
            indexes = {}
            for row in index_rows:
                idx_name = row[0]
                if idx_name not in indexes:
                    indexes[idx_name] = {
                        "name": idx_name,
                        "unique": row[2] == 0,
                        "columns": [],
                    }
                indexes[idx_name]["columns"].append(row[1])
            indexes = list(indexes.values())

            result = {
                "success": True,
                "table": table,
                "database": db_name,
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
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
    try:
        db_name = database or os.getenv("MYSQL_DB", "mysql")
        conn = get_db_connection(db_name)
        with conn.cursor() as cursor:
            query = f"SELECT * FROM `{table}` LIMIT %s"
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

            result = {
                "success": True,
                "table": table,
                "database": db_name,
                "row_count": len(rows),
                "data": rows,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
