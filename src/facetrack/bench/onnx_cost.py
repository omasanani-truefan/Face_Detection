"""Parameter and multiply-accumulate counts read off an ONNX graph at a given input size.

Counts Conv / ConvTranspose / Gemm / MatMul only -- the layers that dominate a CNN
detector's cost; element-wise ops, resizes and activations are ignored. FLOPs are
reported as 2 x MACs, the convention papers use.
"""

from __future__ import annotations

import numpy as np


def onnx_cost(model_path: str, input_hw: tuple[int, int]) -> dict:
    import onnx
    from onnx import shape_inference

    m = onnx.load(model_path)
    inp = m.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    dims[2].ClearField("dim_param")
    dims[2].dim_value = input_hw[0]
    dims[3].ClearField("dim_param")
    dims[3].dim_value = input_hw[1]
    # Graphs exported with intermediate shapes baked in would contradict the new size.
    del m.graph.value_info[:]
    for o in m.graph.output:
        o.type.tensor_type.shape.Clear()
    m = shape_inference.infer_shapes(m)

    shapes: dict[str, list[int]] = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    inits = {i.name: list(i.dims) for i in m.graph.initializer}
    params = int(sum(int(np.prod(d)) for d in inits.values()))

    macs = 0
    for n in m.graph.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            w = inits.get(n.input[1])
            out = shapes.get(n.output[0])
            if not w or not out or 0 in out:
                continue
            # w = [Cout, Cin/groups, kH, kW]; every output element costs Cin/g * kH * kW MACs.
            per_out = int(np.prod(w[1:]))
            macs += int(np.prod(out)) * per_out
        elif n.op_type in ("Gemm", "MatMul"):
            a = shapes.get(n.input[0]) or inits.get(n.input[0])
            out = shapes.get(n.output[0])
            if a and out and 0 not in out:
                macs += int(np.prod(out)) * a[-1]
    return {"params": params, "macs": macs, "gflops": 2 * macs / 1e9, "input_hw": list(input_hw)}
