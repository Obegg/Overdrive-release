#!/usr/bin/env python3
"""Re-saves gtcrn_simple.onnx (sha256 e77603ac...b534) at ORT's extended level: Android arm64
ORT rejects the original graph above level 0, the re-saved one is numerically identical.
Usage: optimize_gtcrn.py gtcrn_simple.onnx gtcrn_simple.opt.onnx"""
import sys

import onnxruntime as ort

src, dst = sys.argv[1], sys.argv[2]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
so.optimized_model_filepath = dst
ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
print("wrote", dst, "with onnxruntime", ort.__version__)
