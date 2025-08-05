# CVAT Serverless Function Modifications for YOLOv8

This document outlines the custom scripts and workflow created to automate the deployment of multiple YOLOv8 models as serverless functions within CVAT.

## Summary

A new system has been implemented to streamline the process of deploying custom YOLOv8 models. Instead of manually creating configuration files for each model, a single script now handles the entire process. This script finds all model files in a specified directory, creates a unique and fully configured serverless function for each one, and deploys them automatically.

**The system now automatically handles both object detection and instance segmentation models.**

## Key Components

### 1. Automated Deployment Script

- **File:** `/home/ubuntu/cvat/serverless/pytorch/ultralytics/deploy_multiple_models.py`
- **Purpose:** This is the main script that drives the automated deployment process.
- **Functionality:**
    - Scans a directory for all `.pt` model files.
    - For each model, it automatically:
        1. Creates a new, unique serverless function directory.
        2. Copies a standard template for the function code (`main.py`, `model_handler.py`).
        3. Loads the model to extract its class names.
        4. Generates a `function.yaml` file, naming the function after the model and populating the class labels automatically.
        5. Calls the standard `deploy_cpu.sh` script to deploy the newly created function to Nuclio.

### 2. Generic Function Template

- **Directory:** `/home/ubuntu/cvat/serverless/pytorch/ultralytics/yolov8/`
- **Purpose:** This directory serves as a template for all new YOLOv8 functions created by the deployment script.
- **Key Files:**
    - `nuclio/model_handler.py`: This has been modified to be a universal handler.
        - It automatically finds and loads whichever `.pt` file is in its directory.
        - **It auto-detects the model type.** If a segmentation model is used, it returns polygons for masks. If a detection model is used, it returns rectangles for bounding boxes.
        - It contains a section for custom label remapping.
    - `nuclio/main.py`: The standard entrypoint for the function, which calls the model handler.
    - `nuclio/function.yaml`: The base configuration that is customized by the deployment script for each new model. It is now configured to deploy functions to the correct `cvat_cvat` Docker network.

## How to Use

The workflow for deploying new models is now much simpler.

### Step 1: Place Model Files

Add all your custom `.pt` model files into the default models directory:

- **`/home/ubuntu/cvat/my_models/`**

This works for both standard detection models (e.g., `yolov8s.pt`) and instance segmentation models (e.g., `yolov8s-seg.pt`).

### Step 2: Run the Deployment Script

Execute the main deployment script from the terminal. **It is critical to use the Python executable from the virtual environment** to ensure all dependencies are found.

```bash
/home/ubuntu/cvat/.venv/bin/python /home/ubuntu/cvat/serverless/pytorch/ultralytics/deploy_multiple_models.py
```

The script will find all models in the default directory, create a unique serverless function for each, and deploy them. You can monitor the progress in the terminal. After the script finishes, the new models will be available as detectors in the CVAT user interface.

## Customization: Remapping Labels

If you want to change a label from your model (e.g., map the model's `bowl` output to a `plate` label in CVAT), you must make the change in two places to ensure the UI and the function output match.

1.  **Deployment Script (for the UI):**
    - **File:** `deploy_multiple_models.py`
    - **Logic:** Add your remapping rule to the `class_names` list. This ensures the `function.yaml` tells the CVAT UI to expect the new label.
    ```python
    # Inside the for loop in deploy_multiple_models.py
    class_names = list(model.names.values())
    for x in range(len(class_names)):
        if class_names[x] in ['bowl', 'donut']:
            class_names[x] = 'plate'
    ```

2.  **Model Handler (for the function's output):**
    - **File:** `yolov8/nuclio/model_handler.py` (the template)
    - **Logic:** Add the same remapping rule inside the `infer` method. This ensures the function's JSON output matches what the UI expects.
    ```python
    # Inside the infer method in model_handler.py
    label = self.model.names[class_id]
    if label in ['bowl', 'donut']:
        label = 'plate'
    ```
After making changes to the template or the deployment script, simply re-run the deployment command from Step 2.
