import os
import json
import subprocess
import sys
from pathlib import Path

def run_command(command_list):
    """Runs a shell command and handles errors."""
    try:
        print(f"Running: {' '.join(command_list)}")
        process = subprocess.run(command_list, capture_output=True, text=True, check=True, cwd=Path.cwd())
        print(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command_list)}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: The command '{command_list[0]}' was not found. Is it installed and in your PATH?", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return False

def setup_vscode_uv_environment():
    project_root = Path.cwd()
    venv_name = ".venv"
    venv_path = project_root / venv_name

    print(f"--- Ensuring `uv` is available ---")
    if not run_command(["uv", "--version"]):
        print("`uv --version` failed. Please ensure `uv` is installed and accessible.", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Step 1: Creating `uv` virtual environment at '{venv_path}' ---")
    if not run_command(["uv", "venv", venv_name]):
        print("Failed to create `uv` virtual environment.", file=sys.stderr)
        sys.exit(1)

    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        print(f"\n--- Step 2: Installing dependencies from '{requirements_file}' ---")
        if not run_command(["uv", "pip", "install", "-r", str(requirements_file)]):
            print("Failed to install dependencies from requirements.txt.", file=sys.stderr)
            # Continue, as the environment is still useful
    else:
        print(f"\n--- Step 2: No '{requirements_file}' found. Skipping dependency installation. ---")
        print("You can install packages later using: uv pip install <package_name>")

    print("\n--- Step 3: Configuring VS Code ---")
    vscode_dir = project_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True)

    settings_file = vscode_dir / "settings.json"

    if sys.platform == "win32":
        interpreter_path = venv_path / "Scripts" / "python.exe"
    else:
        interpreter_path = venv_path / "bin" / "python"

    # Ensure the interpreter executable exists
    if not interpreter_path.exists():
        print(f"Error: Python interpreter not found at expected path: {interpreter_path}", file=sys.stderr)
        print("The `uv venv` command might not have completed successfully or the venv structure is unexpected.", file=sys.stderr)
        sys.exit(1)

    absolute_interpreter_path = str(interpreter_path.resolve())

    settings_data = {}
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                content = f.read()
                if content.strip(): # Check if file is not empty
                    settings_data = json.loads(content)
                else:
                    print(f"Warning: '{settings_file}' was empty. Initializing new settings.")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing '{settings_file}'. It will be overwritten with interpreter settings.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error reading '{settings_file}': {e}. It might be overwritten.", file=sys.stderr)


    settings_data["python.defaultInterpreterPath"] = absolute_interpreter_path

    try:
        with open(settings_file, 'w') as f:
            json.dump(settings_data, f, indent=4)
        print(f"VS Code interpreter path set in '{settings_file}' to: {absolute_interpreter_path}")
    except Exception as e:
        print(f"Error writing to '{settings_file}': {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Setup Complete ---")
    print("1. `uv` virtual environment created (or verified) in '.venv'.")
    print("2. Dependencies from 'requirements.txt' installed (if found).")
    print("3. VS Code 'python.defaultInterpreterPath' setting configured.")
    print("\nTo use the VS Code Python debugger:")
    print("1. Make sure the Python extension is installed in VS Code.")
    print("2. Open your project folder in VS Code. It should now automatically use the `uv` environment.")
    print("3. Open a Python file and start debugging (e.g., by pressing F5 or using the Run and Debug panel).")
    print("   VS Code's default Python debugger settings should work with the selected interpreter.")

if __name__ == "__main__":
    setup_vscode_uv_environment()