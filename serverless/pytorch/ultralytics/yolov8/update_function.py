import argparse
import json
import os
import shutil
import yaml
from ultralytics import YOLO

# Helper to represent literal block strings in pyyaml
class literal_str(str): pass

def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

yaml.add_representer(literal_str, literal_presenter)

def main():
    parser = argparse.ArgumentParser(description="Update Nuclio function with a new YOLOv8 model.")
    parser.add_argument("model_path", type=str, help="Path to the new .pt model file.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # The script is in .../yolov8/, so nuclio is in a subdirectory
    function_dir = os.path.dirname(os.path.realpath(__file__))
    nuclio_dir = os.path.join(function_dir, "nuclio")
    function_yaml_path = os.path.join(nuclio_dir, "function.yaml")

    if not os.path.exists(function_yaml_path):
        print(f"Error: function.yaml not found at {function_yaml_path}")
        return

    print(f"Loading model from {args.model_path}...")
    model = YOLO(args.model_path)

    # In ultralytics, model.names is a dict {index: name}
    class_names = model.names.values()
    print(f"Extracted class names: {list(class_names)}")

    # Create the spec for function.yaml
    spec = [{"name": name, "id": name} for name in class_names]
    spec_json_string = json.dumps(spec, indent=2)

    print(f"Updating {function_yaml_path}...")
    with open(function_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the spec with the new labels, preserving the literal block style
    config['metadata']['annotations']['spec'] = literal_str(spec_json_string)

    with open(function_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"{function_yaml_path} updated successfully.")

    # Copy new model file and remove old one
    for f in os.listdir(nuclio_dir):
        if f.endswith(".pt"):
            old_model_path = os.path.join(nuclio_dir, f)
            print(f"Removing old model file: {old_model_path}")
            os.remove(old_model_path)

    model_filename = os.path.basename(args.model_path)
    destination_path = os.path.join(nuclio_dir, model_filename)
    print(f"Copying {model_filename} to {nuclio_dir}/")
    shutil.copy(args.model_path, destination_path)

    print("\nUpdate complete. You can now redeploy the function:")
    print(f"cd /home/ubuntu/cvat/serverless")
    print(f"./deploy_cpu.sh {os.path.relpath(function_dir, '/home/ubuntu/cvat/serverless')}")


if __name__ == "__main__":
    main()
