from __future__ import annotations

import onnxruntime as ort


def make_session(model_path: str, num_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    # CPU only by design: the target devices have no discrete GPU, and a number measured
    # on CUDA here would say nothing about a phone.
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
