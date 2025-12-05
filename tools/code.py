import os
import subprocess
import tempfile
from pathlib import Path

from langgraph.config import get_stream_writer

from config import Config


def use_python(code: str, timeout: int = 30, save_file: bool = False) -> str:
    """
    Execute Python code and return the result.

    Args:
        code: Python code to execute (can be multi-line)
        timeout: Maximum execution time in seconds (default: 30)
        save_file: Whether to save the code to a file before execution (default: False)

    Returns:
        Execution result with stdout, stderr, and status information
    """
    print(">>> tool message: use_python", code[:100] + "..." if len(code) > 100 else code)
    writer = get_stream_writer()

    # 使用配置中的 Python 路径，如果不存在则使用系统 Python
    python_path = Config.PYTHON_PATH
    if not os.path.exists(python_path):
        python_path = "python3"
        writer(f"Warning: Using system Python ({python_path}) instead of configured path")

    writer("准备执行 Python 代码...")
    writer(f"代码长度: {len(code)} 字符")

    # 确保临时目录存在
    script_dir = None
    if save_file:
        script_dir = Path(Config.PYTHON_CODE_PATH)
        script_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 如果代码包含多行或需要保存文件，使用临时文件执行
        use_file = save_file or "\n" in code or len(code) > 500

        if use_file:
            # 创建临时文件
            temp_dir = str(script_dir) if script_dir else None
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                dir=temp_dir,
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                writer(f"正在执行代码文件: {temp_file}")
                result = subprocess.run(
                    [python_path, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,  # 不自动抛出异常，手动处理
                )
            finally:
                # 如果不保存文件，执行后删除临时文件
                if not save_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
                elif save_file:
                    writer(f"代码已保存到: {temp_file}")
        else:
            # 单行代码直接执行
            writer("正在执行代码...")
            result = subprocess.run(
                [python_path, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        # 构建返回结果
        output_parts = []

        if result.returncode == 0:
            output_parts.append("✅ 代码执行成功")
            if result.stdout:
                writer(f"输出:\n{result.stdout}")
                output_parts.append(f"\n输出:\n{result.stdout}")
            else:
                output_parts.append("\n(无输出)")
        else:
            output_parts.append("❌ 代码执行失败")
            if result.stderr:
                writer(f"错误信息:\n{result.stderr}")
                output_parts.append(f"\n错误信息:\n{result.stderr}")
            if result.stdout:
                output_parts.append(f"\n标准输出:\n{result.stdout}")
            output_parts.append(f"\n返回码: {result.returncode}")

        return "".join(output_parts)

    except subprocess.TimeoutExpired:
        error_msg = f"⏱️ 代码执行超时（超过 {timeout} 秒）"
        writer(error_msg)
        return error_msg

    except FileNotFoundError:
        error_msg = f"❌ 找不到 Python 解释器: {python_path}"
        writer(error_msg)
        return error_msg

    except Exception as ex:
        error_msg = f"❌ 执行过程中发生意外错误: {str(ex)}"
        writer(error_msg)
        return error_msg
