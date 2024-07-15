# TinyLlama Colorist

This project fine-tunes a TinyLlama model to provide color hex codes based on descriptions and integrates the model with Intel® OpenVINO™ for efficient inference.

## Setup

1. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Prepare the data:
    ```bash
    python data/data_preparation.py
    ```

3. Fine-tune the model:
    ```bash
    python models/finetune_model.py
    ```

4. Export the model to ONNX format:
    ```bash
    python models/export_to_onnx.py
    ```

5. Convert the ONNX model to OpenVINO IR format:
    ```bash
    mo --input_model model.onnx --output_dir openvino_model --data_type FP16
    ```

6. Run inference using OpenVINO Inference Engine:
    ```bash
    python models/run_inference_openvino.py
    ```

## License
This project is licensed under the MIT License.
