import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler(context.logger)
    context.user_data.model_handler = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)

    # Get the confidence threshold from the request, or use a default
    threshold = float(data.get("threshold", 0.5))

    # The issue is here: it was calling a different method.
    # We are now calling the correct 'infer' method.
    results = context.user_data.model_handler.infer(image, threshold)

    return context.Response(body=json.dumps(results),
                            headers={},
                            content_type='application/json',
                            status_code=200)
