from pathlib import Path
import numpy as np

from hailo_sdk_client import ClientRunner


def _compile_onnx_to_hef(
    onnx_path,
    calibration_path,
    output_dir,
    model_name,
    hw_arch="hailo10h",
    model_script=None,
):
    """
    ONNX -> parsed HAR -> quantized HAR -> HEF.
    """

    onnx_path = Path(onnx_path)
    calibration_path = Path(calibration_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_har = output_dir / f"{model_name}_parsed.har"
    quantized_har = output_dir / f"{model_name}_quantized.har"
    compiled_har = output_dir / f"{model_name}_compiled.har"
    hef_path = output_dir / f"{model_name}.hef"

    print(f"[HAILO] Compiling {model_name}")
    print(f"[HAILO] ONNX: {onnx_path}")
    print(f"[HAILO] Calibration: {calibration_path}")
    print(f"[HAILO] Target: {hw_arch}")

    # ---------------------------------------------------------
    # 1. Parser
    # Equivalent conceptually to:
    #     hailo parser onnx model.onnx ...
    # ---------------------------------------------------------

    runner = ClientRunner(hw_arch=hw_arch)

    print("[HAILO] Parsing ONNX model...")

    runner.translate_onnx_model(
        str(onnx_path),
        model_name,
    )

    runner.save_har(str(parsed_har))

    print(f"[HAILO] Parsed HAR: {parsed_har}")

    # ---------------------------------------------------------
    # 2. Optional model script
    # ---------------------------------------------------------

    if model_script:
        print(f"[HAILO] Loading model script: {model_script}")
        runner.load_model_script(model_script)

    # ---------------------------------------------------------
    # 3. Optimization / quantization
    #
    # Your calibration .npy should contain representative
    # input samples in the model's input format.
    # ---------------------------------------------------------

    print("[HAILO] Loading calibration data...")

    calibration_data = np.load(
        calibration_path,
        allow_pickle=False,
    )

    print(
        f"[HAILO] Calibration data shape: "
        f"{calibration_data.shape}"
    )

    print("[HAILO] Optimizing / quantizing model...")

    runner.optimize(calibration_data)

    runner.save_har(str(quantized_har))

    print(f"[HAILO] Quantized HAR: {quantized_har}")

    # ---------------------------------------------------------
    # 4. Compiler
    #
    # runner.compile() returns the binary HEF contents.
    # ---------------------------------------------------------

    print("[HAILO] Compiling HEF...")

    hef = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef)

    # Saving after compilation is useful for profiling/debugging.
    runner.save_har(str(compiled_har))

    print(f"[HAILO] Compiled HAR: {compiled_har}")
    print(f"[HAILO] HEF: {hef_path}")

    return hef_path
