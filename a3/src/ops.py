import os
import tvm
from tvm import te
import numpy as np


def make_conv1d_cpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    Apad = te.compute(
        (M + 2 * (N - 1),),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.all(n >= (N - 1), n < (M + N - 1)),
            A[n - N + 1],
            tvm.tir.const(0.0, "float32"),
        ),
        name="Apad",
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            Apad[n + (N - 1) - k] * W[k], axis=k
        ),
        name="B",
    )

    s = te.create_schedule(B.op)
    outer, inner = s[B].split(B.op.axis[0], factor=4)
    s[B].parallel(inner)
    s[B].vectorize(inner)
    return s, A, W, B


def make_conv1d_gpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    Apad = te.compute(
        (M + 2 * (N - 1),),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.all(n >= (N - 1), n < (M + N - 1)),
            A[n - N + 1],
            tvm.tir.const(0.0, "float32"),
        ),
        name="Apad",
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            Apad[n + (N - 1) - k] * W[k], axis=k
        ),
        name="B",
    )
    s = te.create_schedule(B.op)
    outer, inner = s[B].split(B.op.axis[0], factor=32)
    s[B].bind(outer, te.thread_axis("blockIdx.x"))
    s[B].bind(inner, te.thread_axis("threadIdx.x"))
    outerA, innerA = s[Apad].split(Apad.op.axis[0], factor=32)
    s[Apad].bind(outerA, te.thread_axis("blockIdx.x"))
    s[Apad].bind(innerA, te.thread_axis("threadIdx.x"))

    return s, A, W, B


def make_gemm_gpu_scheduler(M, K, N):
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")

    bn = 32
    kfactor = 4
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    s = te.create_schedule(C.op)
    outer_C, inner_C = s[C].split(C.op.axis[0], factor=8)
    s[C].bind(outer_C, te.thread_axis("blockIdx.x"))
    s[C].bind(inner_C, te.thread_axis("threadIdx.x"))
    outer_C2, inner_C2 = s[C].split(C.op.axis[1], factor=8)
    s[C].bind(outer_C2, te.thread_axis("blockIdx.y"))
    s[C].bind(inner_C2, te.thread_axis("threadIdx.y"))
    return s, A, B, C

def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")
    inp_pad = te.compute(
        (B, C, H + (K - 1), W + (K - 1),),
        lambda b, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= (K - 1), w >= (K - 1), ),
            inp[b, c, h - K + 1, w - K + 1],
            tvm.tir.const(0.0, "float32"),
        ),
        name="inp_pad",
    )
    rkh = te.reduce_axis((0, K), name='rkh')
    rkw = te.reduce_axis((0, K), name='rkw')
    out = te.compute(
        (B, C, H, W),
        lambda b, c, i, j: te.sum(
            (inp_pad[b, c, i+rkh, j+rkw] * ker[c, 0, rkh, rkw]),
            axis=[rkh, rkw]), name='Y')
    s = te.create_schedule(out.op)
    outer_C, inner_C = s[out].split(out.op.axis[2], factor=4)
    s[out].parallel(outer_C)
    s[out].parallel(inner_C)
    s[out].bind(outer_C, te.thread_axis("blockIdx.x"))
    s[out].bind(inner_C, te.thread_axis("threadIdx.x"))
    outer_C, inner_C = s[out].split(out.op.axis[3], factor=4)
    s[out].parallel(outer_C)
    s[out].parallel(inner_C)
    s[out].bind(outer_C, te.thread_axis("blockIdx.z"))
    s[out].bind(inner_C, te.thread_axis("threadIdx.z"))
    outer_inp_pad, inner_inp_pad = s[inp_pad].split(inp_pad.op.axis[2], factor=4)
    s[out].parallel(outer_C)
    s[out].parallel(inner_C)
    s[inp_pad].bind(outer_inp_pad, te.thread_axis("blockIdx.y"))
    s[inp_pad].bind(inner_inp_pad, te.thread_axis("threadIdx.y"))
    return s, inp, ker, out
