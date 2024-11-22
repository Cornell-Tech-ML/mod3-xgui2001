# type: ignore
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
        else:
            # index for output tensor
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # position in input tensor
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

        # apply function to input tensor and store in output tensor
        out[out_pos] = fn(in_storage[in_pos])

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
        else:
            # index for output tensor
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)

            a_pos = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

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

    # Load data into shared memory if thread is within bounds
    if i < size:
        cache[pos] = a[i]

    # Thread 0 performs the reduction for its block
    if pos == 0:
        # Initialize accumulator for block sum
        block_sum = 0

        # Calculate how many elements are in this block
        elements_in_block = min(BLOCK_DIM, size - i)

        # Sum all elements in the block
        for j in range(elements_in_block):
            block_sum += cache[j]

        # Store block result in output array
        out[cuda.blockIdx.x] = block_sum


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
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Calculate dimensions and sizes
        dimension_size = out_shape[reduce_dim]  # Size of dimension being reduced
        remaining_dimensions_product = (
            out_size // dimension_size
        )  # Product of other dimensions
        current_block_idx = (
            out_pos // remaining_dimensions_product
        )  # Current reduction block

        if current_block_idx * BLOCK_DIM + pos < a_shape[reduce_dim]:
            # Convert flat output position to multi-dimensional index
            to_index(out_pos, out_shape, out_index)
            out_index[reduce_dim] = pos
            a_index = index_to_position(out_index, a_strides)
            cache[pos] = a_storage[a_index]

        cuda.syncthreads()

        if pos == 0:
            acc = reduce_value
            reduce_size = min(
                BLOCK_DIM,  # Either process full block
                a_shape[reduce_dim]
                - BLOCK_DIM * current_block_idx
                - pos,  # Or remaining elements
            )
            for i in range(reduce_size):
                acc = fn(acc, cache[i])
            out[out_pos] = acc

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
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Check thread boundaries before computation
    if cuda.threadIdx.x < size and cuda.threadIdx.y < size:
        # Load matrices into shared memory
        # Each thread loads one element from each input matrix
        # Using row-major layout: index = row * width + column
        shared_a[cuda.threadIdx.x, cuda.threadIdx.y] = a[
            cuda.threadIdx.x * size + cuda.threadIdx.y
        ]

        shared_b[cuda.threadIdx.x, cuda.threadIdx.y] = b[
            cuda.threadIdx.x * size + cuda.threadIdx.y
        ]

    # Synchronize to ensure all threads have finished loading data
    cuda.syncthreads()

    # Initialize accumulator for dot product
    acc = 0

    # Compute dot product for this thread's output element
    # Each thread computes one element of result matrix
    for k in range(size):
        # Multiply and accumulate corresponding elements
        # from matrix A's row and matrix B's column
        acc += shared_a[cuda.threadIdx.x, k] * shared_b[k, cuda.threadIdx.y]

    # Store final result in output matrix using linear indexing
    out[cuda.threadIdx.x * size + cuda.threadIdx.y] = acc


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

    # Initialize accumulator for the dot product result for this thread
    # Each thread computes exactly one element of the output matrix
    value = 0.0

    # Iterate over matrix chunks of size BLOCK_DIM x BLOCK_DIM
    # Using ceiling division to handle matrices whose dimensions aren't multiples of BLOCK_DIM
    for tile in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load data from matrix A into shared memory if the thread's position is valid
        # This check ensures we don't access memory outside the matrix boundaries
        if i < a_shape[-2] and tile * BLOCK_DIM + pj < a_shape[-1]:
            # Calculate the exact position in global memory for matrix A element
            # batch * a_batch_stride: moves to the correct batch if batched multiplication
            # i * a_strides[-2]: moves to the correct row
            # (tile * BLOCK_DIM + pj) * a_strides[-1]: moves to the correct column within the current tile
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride
                + i * a_strides[-2]
                + (tile * BLOCK_DIM + pj) * a_strides[-1]
            ]
        else:
            # If thread position is outside matrix bounds, pad with zero
            # This ensures correct computation without affecting the result
            a_shared[pi, pj] = 0.0

        # Similar boundary check for matrix B
        # Validates if the thread's position falls within matrix B's dimensions
        if tile * BLOCK_DIM + pi < b_shape[-2] and j < b_shape[-1]:
            # Calculate global memory position for matrix B element
            # batch * b_batch_stride: batch offset for batched operations
            # (tile * BLOCK_DIM + pi) * b_strides[-2]: row offset within current tile
            # j * b_strides[-1]: column offset
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride
                + (tile * BLOCK_DIM + pi) * b_strides[-2]
                + j * b_strides[-1]
            ]
        else:
            # Zero padding for out-of-bounds positions
            # Maintains correct computation without contributing to the result
            b_shared[pi, pj] = 0.0

        # Synchronization barrier - ensures all threads have finished loading data
        # Critical for preventing race conditions and maintaining data consistency
        cuda.syncthreads()

        # Compute partial dot product for current tile
        # Iterates through corresponding elements in the shared memory tiles
        for k in range(BLOCK_DIM):
            # Accumulate the product of corresponding elements
            # a_shared[pi, k] is element from matrix A's row
            # b_shared[k, pj] is element from matrix B's column
            value += a_shared[pi, k] * b_shared[k, pj]

        # Synchronization barrier before next iteration
        # Ensures all threads complete computation before loading next tile
        cuda.syncthreads()

    # Final boundary check before writing result
    # Ensures thread is computing a valid element of output matrix
    if i < out_shape[-2] and j < out_shape[-1]:
        # Calculate flattened index in output array using strides
        # batch * out_strides[0]: moves to correct batch
        # i * out_strides[1]: moves to correct row
        # j * out_strides[2]: moves to correct column
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        # Store computed result in global memory
        out[out_pos] = value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
