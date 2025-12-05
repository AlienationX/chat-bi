import json
import os
from typing import Optional

import psycopg2
from mcp.server.fastmcp import FastMCP
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

mcp = FastMCP("PostgreSQL Database")


def get_db_connection():
    """获取 PostgreSQL 数据库连接"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "124.222.205.137"),
            port=os.getenv("POSTGRES_PORT", "5434"),
            database=os.getenv("POSTGRES_DB", "samples"),
            user=os.getenv("POSTGRES_USER", "samples_user"),
            password=os.getenv("POSTGRES_PASSWORD", "fb:aoPkF?lIt?KJ~V4Kfq=2$Fvmq$K=T3Jac_Ztyq-D8~EU"),
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
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
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
                    "data": [dict(row) for row in rows],
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

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        error_result = {
            "success": False,
            "error": str(e),
            "error_code": e.pgcode if hasattr(e, "pgcode") else None,
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
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database="postgres",  # 连接到默认数据库
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname")
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
async def list_tables(database: Optional[str] = None, schema: str = "public") -> str:
    """
    列出指定数据库和模式中的所有表。

    Args:
        database: 数据库名称（可选，默认使用配置的数据库）
        schema: 模式名称（默认 'public'）

    Returns:
        JSON 格式的表列表
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 如果指定了数据库，切换到该数据库
            if database:
                conn.close()
                conn = psycopg2.connect(
                    host=os.getenv("POSTGRES_HOST", "localhost"),
                    port=os.getenv("POSTGRES_PORT", "5432"),
                    database=database,
                    user=os.getenv("POSTGRES_USER", "postgres"),
                    password=os.getenv("POSTGRES_PASSWORD", ""),
                )
                cursor = conn.cursor()

            query = sql.SQL(
                """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
                """
            )
            cursor.execute(query, [schema])
            tables = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]

            result = {
                "success": True,
                "schema": schema,
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
async def describe_table(table: str, schema: str = "public") -> str:
    """
    获取表的详细结构信息，包括列名、数据类型、约束等。

    Args:
        table: 表名
        schema: 模式名称（默认 'public'）

    Returns:
        JSON 格式的表结构信息
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # 获取列信息
            columns_query = sql.SQL(
                """
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """
            )
            cursor.execute(columns_query, [schema, table])
            columns = [dict(row) for row in cursor.fetchall()]

            # 获取主键信息
            pk_query = sql.SQL(
                """
                SELECT column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_schema = %s 
                    AND tc.table_name = %s
                    AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position
                """
            )
            cursor.execute(pk_query, [schema, table])
            primary_keys = [row[0] for row in cursor.fetchall()]

            # 获取外键信息
            fk_query = sql.SQL(
                """
                SELECT
                    kcu.column_name,
                    ccu.table_schema AS foreign_table_schema,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_schema = %s
                    AND tc.table_name = %s
                """
            )
            cursor.execute(fk_query, [schema, table])
            foreign_keys = [dict(row) for row in cursor.fetchall()]

            # 获取索引信息
            index_query = sql.SQL(
                """
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = %s AND tablename = %s
                """
            )
            cursor.execute(index_query, [schema, table])
            indexes = [{"name": row[0], "definition": row[1]} for row in cursor.fetchall()]

            result = {
                "success": True,
                "table": table,
                "schema": schema,
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
async def get_table_sample(table: str, schema: str = "public", limit: int = 10) -> str:
    """
    获取表的示例数据（前 N 行）。

    Args:
        table: 表名
        schema: 模式名称（默认 'public'）
        limit: 返回的行数（默认 10）

    Returns:
        JSON 格式的示例数据
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(sql.Identifier(schema), sql.Identifier(table))
            cursor.execute(query, [limit])
            rows = [dict(row) for row in cursor.fetchall()]

            result = {
                "success": True,
                "table": table,
                "schema": schema,
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


@mcp.tool()
async def list_schemas(database: Optional[str] = None) -> str:
    """
    列出指定数据库中的所有模式。

    Args:
        database: 数据库名称（可选，默认使用配置的数据库）

    Returns:
        JSON 格式的模式列表
    """
    conn = None
    try:
        if database:
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=database,
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
            )
        else:
            conn = get_db_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                ORDER BY schema_name
                """
            )
            schemas = [row[0] for row in cursor.fetchall()]

            result = {
                "success": True,
                "schemas": schemas,
                "count": len(schemas),
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
