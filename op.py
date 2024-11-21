from tvm import te
from tvm import relax


def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")


def te_relu(A:te.Tensor)->te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")