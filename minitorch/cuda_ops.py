# pytorch cuda ops
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to compile a function for CUDA.

    Args:
    ----
        fn (Fn): function to compile.
        **kwargs: keyword arguments for numba.jit.

    Returns:
    -------
        Fn: compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to compile a function for CUDA.

    Args:
    ----
        fn: function to compile.
        **kwargs: keyword arguments for numba.jit.

    Returns:
    -------
        FakeCUDAKernel: compiled function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """See `tensor_ops.py`"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Only process if thread is within output size
        if i >= out_size:
            return

        # index for output tensor
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)

        # position in input tensor
        in_pos = index_to_position(in_index, in_strides)

        # apply function to input tensor and store in output tensor
        out[i] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Only process if thread is within output size
        if i >= out_size:
            return

        # index for input tensor
        to_index(i, out_shape, out_index)

        # flat indices for input and ouput
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # position in input and output tensors
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)

        # apply function to input tensors and store in output tensor
        out[i] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # only process if thread is within input size
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0

    cuda.syncthreads()

    # Perform reduction within the block
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride *= 2

    # Write the result for this block to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice sum function.

    Args:
    ----
        a (Tensor): input tensor.

    Returns:
    -------
        TensorData: output tensor.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # BLOCK_DIM = 1024
        # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        # block_pos = cuda.blockIdx.x
        # thread_pos = cuda.threadIdx.x

        # Global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Only process if thread is within output size
        if i < out_size:
            # index for output tensor
            to_index(i, out_shape, out_index)
            # position in output tensor
            out_pos = index_to_position(out_index, out_strides)

            for j in range(a_shape[reduce_dim]):
                # index for input tensor
                out_index[reduce_dim] = j
                # position in input tensor
                in_pos = index_to_position(out_index, a_strides)
                # apply function to input tensor and store in output tensor
                reduce_value = fn(reduce_value, a_storage[in_pos])

            out[out_pos] = reduce_value

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # Create tile buffers in shared memory
    tile_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    tile_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)

    # Get local thread coordinates
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Skip threads outside matrix dimensions
    if i >= size or j >= size:
        return

    # Copy input values to shared memory buffers
    tile_a[i, j] = a[size * i + j]
    tile_b[i, j] = b[size * i + j]

    # Wait for all threads to finish loading
    cuda.syncthreads()

    # Calculate matrix product element
    sum_val = 0.0
    for idx in range(size):
        sum_val += tile_a[i, idx] * tile_b[idx, j]

    # Store final value in output matrix
    out[size * i + j] = sum_val


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice matmul function.

    Args:
    ----
        a (Tensor): input tensor.
        b (Tensor): input tensor.

    Returns:
    -------
        TensorData: output tensor.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # 'value' is the accumulator for the dot product of the corresponding row of a and column of b.
    # Each thread will compute one elemnet of the result matrix.
    value = 0.0

    # This loop moves thorugh the shred dimension (columsn of a and rows of b)  in size BLOCK_DIM.
    # THe loop ensures that all parts of the matrices are processesd, even if the matric's dimension arent
    # exact multiples of BLOCK_DIM.
    for tile in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # Check if thread is within bounds of the matric deimnsions and load the corresponding
        # element from global memory (a_storage) into the shared memory (a_shared).
        # In the case the thread is out of bounds (e.g. when the matrix
        # dimensions do not perfectly align with block size), assign 0.0 to shared memory to pad the tile.
        # (as multiplying by 0 does not affect the final result)
        if i < a_shape[-2] and tile * BLOCK_DIM + pj < a_shape[-1]:
            # Calculate the global memory address for element of a:
            # 1. batch * a_batch_stride: batch offset (if batching is used)
            # 2. row * a_strides[-2]: row offset
            # 3. (tile * BLOCK_DIM + thread_y) * a_strides[-1]: column offset
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride
                + i * a_strides[-2]
                + (tile * BLOCK_DIM + pj) * a_strides[-1]
            ]
        else:
            a_shared[pi, pj] = 0.0

        # Do the same thing for b (Checking bounds and loading into shared memory)
        if tile * BLOCK_DIM + pi < b_shape[-2] and j < b_shape[-1]:
            # Calculate the global memory address for element of b:
            # 1. batch * b_batch_stride: batch offset (if batching is used)
            # 2. (tile * BLOCK_DIM + pi) * b_strides[-2]: row offset
            # 3. j * b_strides[-1]: column offset
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride
                + (tile * BLOCK_DIM + pi) * b_strides[-2]
                + j * b_strides[-1]
            ]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize threads to ensure tiles are fully loaded before proceding with computation.
        cuda.syncthreads()

        # Each thread computes part of the dot product for its assigned row-column pair using the
        # shared memroy tiles. The k loop iterates over the elements of the shared_memory tiles
        for k in range(BLOCK_DIM):
            value += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize threads before loading the next tile
        cuda.syncthreads()

    # Check if the threads' position is within bounds of the output matrix and write the result to the
    # output matrix. Calculate the flattened index (out_pos) for the output tensor using strides. Write the computed
    # value back to global memory.
    if i < out_shape[-2] and j < out_shape[-1]:
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_pos] = value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
