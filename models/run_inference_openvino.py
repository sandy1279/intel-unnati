import numpy as np
import openvino.runtime as ov
from transformers import AutoTokenizer

model_path = "openvino_model/model.xml"
core = ov.Core()
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

def generate_response(user_input):
    prompt = formatted_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].numpy()

    results = compiled_model([input_ids])
    logits = results[output_layer]
    
    generated_tokens = np.argmax(logits, axis=-1)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_text
