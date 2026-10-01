#!/usr/bin/env python3
import time 
import os
from io import open
import sys
from pathlib import Path
import pandas as pd
import argparse
import ctypes
import socket
import pickle

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gpt_model import GPTSmartMeterModel
from utils.universal_tokeniser_naive import load_tokenisers_as_vocab

SERVER_INET_PORT = 8888

SEQUENCE_LENGTH = 336
N_FEATURES = 11
D_MODEL = 64

SAMPLE_SIZE_BYTES = N_FEATURES * SEQUENCE_LENGTH * 4

HB_DEMO_OUTPUT = 0
HB_DEMO_DEBUG_TOKENS = 1 << 0
HB_DEMO_DEBUG_EMBEDDINGS = 1 << 1
HB_DEMO_DEBUG_QUANTIZED_INPUT = 1 << 2
HB_DEMO_DEBUG_QUANTIZED_OUTPUT = 1 << 3
HB_DEMO_DEBUG_DEQUANTIZED_OUTPUT = 1 << 4
HB_DEMO_DEBUG_DECODED_TOKENS = 1 << 5

INPUT_TYPE_RAW = 0
INPUT_TYPE_TOKENS = 1

class InferenceRequest(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("input_type", ctypes.c_uint32),
        ("debug_flags", ctypes.c_uint32),
        ("value", ctypes.c_uint8 * SAMPLE_SIZE_BYTES),
    ]


class InferenceResponse(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("debug_flags", ctypes.c_uint32),
        ("payload_len", ctypes.c_uint32),
    ]


def recv_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError("Server disconnected")

        data.extend(chunk)

    return bytes(data)


def send_request(sock, request_id, raw_values, input_type):
    if len(raw_values) != SAMPLE_SIZE_BYTES:
        raise ValueError(
            f"Expected {SAMPLE_SIZE_BYTES} bytes, "
            f"got {len(raw_values)}"
        )

    req = InferenceRequest()

    req.request_id = request_id
    req.input_type = input_type

    # Ask for everything useful.
    req.debug_flags = (
        HB_DEMO_DEBUG_TOKENS
        | HB_DEMO_DEBUG_EMBEDDINGS
        | HB_DEMO_DEBUG_QUANTIZED_INPUT
        | HB_DEMO_DEBUG_QUANTIZED_OUTPUT
        | HB_DEMO_DEBUG_DEQUANTIZED_OUTPUT
        | HB_DEMO_DEBUG_DECODED_TOKENS
    )

    ctypes.memmove(
        ctypes.addressof(req) + InferenceRequest.value.offset,
        raw_values,
        SAMPLE_SIZE_BYTES,
    )

    sock.sendall(bytes(req))


def receive_response(sock):
    """
    Receive all debug messages for one request.

    Returns:
        dict[str, np.ndarray]
    """

    results = {}

    while True:
        header_data = recv_exact(
            sock,
            ctypes.sizeof(InferenceResponse),
        )

        resp = InferenceResponse.from_buffer_copy(header_data)

        payload = recv_exact(sock, resp.payload_len)

        print(
            f"Response: request_id={resp.request_id} "
            f"flags={resp.debug_flags} "
            f"payload_len={resp.payload_len}"
        )

        if resp.debug_flags == HB_DEMO_OUTPUT:
            results["output"] = np.frombuffer(
                payload,
                dtype=np.float32,
            ).copy()

            break

        elif resp.debug_flags == HB_DEMO_DEBUG_TOKENS:
            results["tokens"] = np.frombuffer(
                payload,
                dtype=np.int32,
            ).copy()

        elif resp.debug_flags == HB_DEMO_DEBUG_EMBEDDINGS:
            results["embeddings"] = np.frombuffer(
                payload,
                dtype=np.float32,
            ).copy()

        elif resp.debug_flags == HB_DEMO_DEBUG_QUANTIZED_INPUT:
            # Change dtype if your Hailo input is not uint8.
            results["quantized_input"] = np.frombuffer(
                payload,
                dtype=np.uint8,
            ).copy()

        elif resp.debug_flags == HB_DEMO_DEBUG_QUANTIZED_OUTPUT:
            # Change dtype if required.
            results["quantized_output"] = np.frombuffer(
                payload,
                dtype=np.uint8,
            ).copy()

        elif resp.debug_flags == HB_DEMO_DEBUG_DEQUANTIZED_OUTPUT:
            results["dequantized_output"] = np.frombuffer(
                payload,
                dtype=np.float32,
            ).copy()

        elif resp.debug_flags == HB_DEMO_DEBUG_DECODED_TOKENS:
            results["decoded_tokens"] = np.frombuffer(
                payload,
                dtype=np.int32,
            ).copy()

        else:
            print(
                f"Unhandled response flag: "
                f"{resp.debug_flags}"
            )

    return results

def tokenize_raw_input(raw, tokeniser_path):
    raw = np.asarray(raw, dtype=np.float32)

    if raw.shape != (SEQUENCE_LENGTH, N_FEATURES):
        raise RuntimeError(
            f"Expected {(SEQUENCE_LENGTH, N_FEATURES)}, got {raw.shape}"
        )

    raw = raw.reshape(
        1,
        SEQUENCE_LENGTH,
        N_FEATURES,
    )

    vocabs = load_tokenisers_as_vocab(
        tokeniser_path
    )

    tokens = np.empty(
        raw.shape,
        dtype=np.int32,
    )

    tokens[:, :, 0] = vocabs["electricity"].encode_array(
        raw[:, :, 0].reshape(-1)
    ).reshape(1, SEQUENCE_LENGTH)

    tokens[:, :, 1] = vocabs["gas"].encode_array(
        raw[:, :, 1].reshape(-1)
    ).reshape(1, SEQUENCE_LENGTH)

    tokens[:, :, 2] = vocabs["temperature"].encode_array(
        raw[:, :, 2].reshape(-1)
    ).reshape(1, SEQUENCE_LENGTH)

    tokens[:, :, 3] = vocabs["solar"].encode_array(
        raw[:, :, 3].reshape(-1)
    ).reshape(1, SEQUENCE_LENGTH)

    tokens[:, :, 4:] = raw[:, :, 4:].astype(np.int32)

    tokens[:, 0, 0] = vocabs["electricity"].sos_token
    tokens[:, 0, 1] = vocabs["gas"].sos_token

    return tokens


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_exact(name, reference, actual):
    reference = np.asarray(reference).reshape(-1)
    actual = np.asarray(actual).reshape(-1)

    print()
    print(f"=== {name} ===")

    if reference.shape != actual.shape:
        print(
            f"SHAPE MISMATCH: "
            f"reference={reference.shape}, "
            f"remote={actual.shape}"
        )
        return False

    mismatch = reference != actual
    count = np.count_nonzero(mismatch)

    if count == 0:
        print(f"PASS: {reference.size} values match exactly")
        return True

    print(
        f"FAIL: {count}/{reference.size} values differ"
    )

    indices = np.flatnonzero(mismatch)

    for index in indices[:10]:
        print(
            f"  [{index}] "
            f"reference={reference[index]} "
            f"remote={actual[index]}"
        )

    return False


def compare_float(
    name,
    reference,
    actual,
    *,
    rtol=1e-5,
    atol=1e-6,
):
    reference = np.asarray(
        reference,
        dtype=np.float32,
    ).reshape(-1)

    actual = np.asarray(
        actual,
        dtype=np.float32,
    ).reshape(-1)

    print()
    print(f"=== {name} ===")

    if reference.shape != actual.shape:
        print(
            f"SHAPE MISMATCH: "
            f"reference={reference.shape}, "
            f"remote={actual.shape}"
        )
        return False

    diff = np.abs(reference - actual)

    close = np.isclose(
        reference,
        actual,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )

    mismatch_count = np.count_nonzero(~close)

    print(f"elements:       {reference.size}")
    print(f"mismatches:     {mismatch_count}")
    print(f"max abs error:  {diff.max():.9g}")
    print(f"mean abs error: {diff.mean():.9g}")

    denominator = np.maximum(
        np.abs(reference),
        1e-12,
    )

    relative_diff = diff / denominator

    print(
        f"max rel error:  "
        f"{relative_diff.max():.9g}"
    )

    if mismatch_count == 0:
        print("PASS")
        return True

    print("FAIL")

    indices = np.flatnonzero(~close)

    print("First mismatches:")

    for index in indices[:10]:
        print(
            f"  [{index}] "
            f"reference={reference[index]:.9g} "
            f"remote={actual[index]:.9g} "
            f"abs_err={diff[index]:.9g}"
        )

    return False


def compare_results(reference, remote):
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if (
        "tokens" in reference
        and "tokens" in remote
    ):
        compare_exact(
            "TOKENS",
            reference["tokens"],
            remote["tokens"],
        )

    if (
        "embeddings" in reference
        and "embeddings" in remote
    ):
        compare_float(
            "EMBEDDINGS",
            reference["embeddings"],
            remote["embeddings"],
            rtol=1e-5,
            atol=1e-6,
        )

    if (
        "output" in reference
        and "output" in remote
    ):
        compare_float(
            "FINAL OUTPUT",
            reference["output"],
            remote["output"],
            rtol=1e-4,
            atol=1e-5,
        )

    if (
        "dequantized_output" in reference
        and "dequantized_output" in remote
    ):
        compare_float(
            "DEQUANTIZED OUTPUT",
            reference["dequantized_output"],
            remote["dequantized_output"],
            rtol=1e-4,
            atol=1e-5,
        )
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "host",
        help="IP address or hostname of server",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_INET_PORT,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model (.pth)",
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to training shard (.pt)",
    )

    args = parser.parse_args()

    print("Running PyTorch reference...")

    # --------------------------------------------------
    # Load training shard
    # --------------------------------------------------

    shard = torch.load(
        args.data_file,
        map_location="cpu",
        weights_only=False,
    )

    X_train = shard["X_train"]

    print(
        "Training data shape:",
        X_train.shape,
    )

    if X_train.ndim != 3:
        raise RuntimeError(
            f"Expected X_train to have 3 dimensions, "
            f"got {X_train.shape}"
        )

    if X_train.shape[1:] != (
        SEQUENCE_LENGTH,
        N_FEATURES,
    ):
        raise RuntimeError(
            f"Expected samples shaped "
            f"{(SEQUENCE_LENGTH, N_FEATURES)}, "
            f"got {X_train.shape[1:]}"
        )

    # --------------------------------------------------
    # Load PyTorch model
    # --------------------------------------------------

    model = torch.load(
        args.model_path,
        map_location="cpu",
        weights_only=False,
    )

    model.eval()

    # --------------------------------------------------
    # Process each training sample
    # --------------------------------------------------

    for sample_index in range(len(X_train)):
        print(
            f"Processing sample "
            f"{sample_index}/{len(X_train)}"
        )

        # X_train sample is already tokenised.
        tokens = X_train[sample_index]

        # Model expects [batch, sequence, features].
        tokens_tensor = tokens.unsqueeze(0)

        print(
            "Token shape:",
            tokens_tensor.shape,
        )

        # --------------------------------------------------
        # PyTorch reference
        # --------------------------------------------------

        with torch.no_grad():
            model(
                tokens_tensor,
                backend="cpu",
                log=True,
            )

        reference = model.get_debug_cache()

        print(
            "Reference results:",
            list(reference.keys()),
        )

        # --------------------------------------------------
        # Remote execution
        # --------------------------------------------------

        tokens_np = (
            tokens
            .detach()
            .cpu()
            .numpy()
            .astype("<i4")
        )

        token_values = tokens_np.tobytes()

        with socket.create_connection(
            (args.host, args.port)
        ) as sock:

            print(
                f"Connected to "
                f"{args.host}:{args.port}"
            )

            send_request(
                sock,
                sample_index,
                token_values,
                INPUT_TYPE_TOKENS,
            )

            remote = receive_response(
                sock
            )

        print(
            "Remote results:",
            list(remote.keys()),
        )

        # --------------------------------------------------
        # Compare
        # --------------------------------------------------

        compare_results(
            reference,
            remote,
        )

        time.sleep(1)


if __name__ == "__main__":
    main()
