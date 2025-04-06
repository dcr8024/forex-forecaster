import subprocess
import sys
import os


def run_script(script_name):
    """Run a Python script and check for success"""
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name],
                            capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in {script_name}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"Successfully completed {script_name}")
        return result.stdout


def main():

    scripts = [
        "forex_price_loader.py",
        "cot_data_loader.py",
        "cot_analyzer.py"
    ]

    # Check if all scripts exist
    for script in scripts:
        if not os.path.exists(script):
            print(f"Error: {script} not found!")
            sys.exit(1)

    # Execute scripts in order
    for script in scripts:
        output = run_script(script)
        print(f"Output: {output}")
        print("-" * 50)


if __name__ == "__main__":
    main()
