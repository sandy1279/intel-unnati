import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "tinyllama-colorist-v1/checkpoint-250"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, offload_buffers=True)
peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")
model = peft_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

dummy_input = tokenizer("Bright red", return_tensors="pt").input_ids.to("cuda")

torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    input_names=["input_ids"], 
    output_names=["logits"], 
    opset_version=11,
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence"}}
)
