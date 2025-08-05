#!/home/ubuntu/cvat/.venv/bin/python

import argparse
import json
import os
import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO
import base64
import requests
import re
# Helper to represent literal block strings in pyyaml
class literal_str(str): pass

def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

yaml.add_representer(literal_str, literal_presenter)

def main():
    parser = argparse.ArgumentParser(
        description="Find all .pt models in a directory, create a Nuclio function for each, and deploy them.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d", "--models_dir",
        type=str,
        default="/home/ubuntu/cvat/my_models",
        help="Path to the directory containing .pt model files (default: %(default)s)"
    )
    args = parser.parse_args()

    models_path = Path(args.models_dir).resolve()
    if not models_path.is_dir():
        print(f"Error: Directory not found at {models_path}", file=sys.stderr)
        sys.exit(1)

    # This script is in .../ultralytics/, the template is in a subdirectory 'yolov8/nuclio'
    script_dir = Path(__file__).parent.resolve()
    template_dir = script_dir / "yolov8" / "nuclio"
    serverless_dir = script_dir.parent.parent # up to serverless/

    if not template_dir.exists() or not (template_dir / "function.yaml").exists():
        print(f"Error: Template directory or function.yaml not found in {template_dir}", file=sys.stderr)
        sys.exit(1)

    # Dynamically import YOLO from ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found. Please install it: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    pt_files = list(models_path.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {models_path}")
        return

    print(f"Found {len(pt_files)} model(s) to deploy...")

    for model_file in pt_files:
        model_name = model_file.stem
        # Sanitize the model name for use in a directory path
        sanitized_model_name = model_name.lower().replace('_', '-').replace('.', '-')
        function_dir_name = f"yolov8-{sanitized_model_name}"
        function_name = f"pth-ultralytics-{function_dir_name}" # Match nuclio naming convention

        print(f"\n--- Processing model: {model_file.name} ---")
        print(f"Function name will be: {function_name}")

        # 1. Create a new directory for the function
        target_function_dir = script_dir / function_dir_name
        target_nuclio_dir = target_function_dir / "nuclio"

        if target_nuclio_dir.exists():
            print(f"Function directory {target_nuclio_dir} already exists. Recreating it.")
            shutil.rmtree(target_nuclio_dir)

        target_nuclio_dir.mkdir(parents=True)
        print(f"Created directory: {target_nuclio_dir}")

        # 2. Copy template files and the new model
        shutil.copy(template_dir / "main.py", target_nuclio_dir)
        shutil.copy(template_dir / "model_handler.py", target_nuclio_dir)
        shutil.copy(model_file, target_nuclio_dir)
        print("Copied function code and model file.")

        # 3. Load model and extract labels
        print("Loading model to extract class names...")
        model = YOLO(str(model_file))
        class_names = list(model.names.values())
        for x in range(len(class_names)):
            class_names[x] = 'plate' if class_names[x] in ['bowl'] else class_names[x]
        print(f"Extracted class names: {class_names}")

        # 4. Create function.yaml
        spec = [{"name": name, "id": name} for name in class_names]
        spec_json_string = json.dumps(spec, indent=2)

        with open(template_dir / "function.yaml", 'r') as f:
            config = yaml.safe_load(f)

        config['metadata']['name'] = function_name
        config['metadata']['annotations']['name'] = f"YOLOv8 {model_name}"
        config['metadata']['annotations']['spec'] = literal_str(spec_json_string)
        config['spec']['description'] = f"YOLOv8 model detector for {model_file.name}"
        config['spec']['build']['image'] = f"cvat.{function_name}"

        function_yaml_path = target_nuclio_dir / "function.yaml"
        with open(function_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created {function_yaml_path}")

        # 5. Deploy the function
        print("Deploying function...")
        deploy_script = serverless_dir / "deploy_cpu.sh"
        # The deploy script needs the relative path from serverless/
        relative_path_for_deploy = target_function_dir.relative_to(serverless_dir)

        process = subprocess.run(
            [str(deploy_script), str(relative_path_for_deploy)],
            cwd=serverless_dir,
            capture_output=True,
            text=True
        )

        if process.returncode == 0:
            print("Deployment successful!")
            if process.stdout: print(f"Output:\n{process.stdout}")

            # # 6. Test the deployed function
            # print(f"\n--- Testing model: {model_file.name} ---")
            # node_port = 0
            # # Try to find the port in the deployment output
            # match = re.search(rf"{function_name}\s+.*?\s+(\d+)\s+", process.stdout)
            # if match:
            #     node_port = int(match.group(1))
            #     print(f"Found function port: {node_port}")

            # if node_port > 0:
            #     test_image_path = "/home/ubuntu/cvat/original_image (3).png"
            #     if not os.path.exists(test_image_path):
            #         print(f"Test image not found at {test_image_path}, skipping test.")
            #     else:
            #         try:
            #             with open(test_image_path, "rb") as f:
            #                 image_data = base64.b64encode(f.read()).decode("utf-8")

            #             data = {"image": image_data}
            #             # sudo ufw allow node_port
            #             os.system(f"sudo ufw allow {node_port}")
            #             url = f"http://0.0.0.0:{node_port}"
            #             print(f"Sending test request to {url}...")
            #             response = requests.post(url, json=data, timeout=300)
            #             response.raise_for_status() # Raise an exception for bad status codes
            #             print("Test Response:")
            #             print(response.json())
            #         except requests.exceptions.RequestException as e:
            #             print(f"Error during test request: {e}", file=sys.stderr)
            #         except Exception as e:
            #             print(f"An unexpected error occurred during testing: {e}", file=sys.stderr)
            # else:
            #     print("Could not determine function port. Skipping test.")

        else:
            print("--- Deployment Failed ---", file=sys.stderr)
            if process.stdout: print(f"STDOUT:\n{process.stdout}", file=sys.stderr)
            if process.stderr: print(f"STDERR:\n{process.stderr}", file=sys.stderr)
            print("-------------------------", file=sys.stderr)

    print("\nAll models processed.")

if __name__ == "__main__":
    main()
