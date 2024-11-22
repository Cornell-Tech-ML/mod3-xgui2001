# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

* 3.1 Diagnostic Output

MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (191)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (191)
--------------------------------------------------------------------------------|loop #ID
    def _map(                                                                   |
        out: Storage,                                                           |
        out_shape: Shape,                                                       |
        out_strides: Strides,                                                   |
        in_storage: Storage,                                                    |
        in_shape: Shape,                                                        |
        in_strides: Strides,                                                    |
    ) -> None:                                                                  |
        # When tensors are aligned                                              |
        if check_arrays_match(out_strides, in_strides, out_shape, in_shape):    |
            for i in prange(len(out)):------------------------------------------| #2
                out[i] = fn(in_storage[i])                                      |
            return                                                              |
        else:                                                                   |
            # When tensors are not aligned                                      |
            for i in prange(len(out)):------------------------------------------| #3
                out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)-------------| #0
                to_index(i, out_shape, out_idx)                                 |
                                                                                |
                in_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------------| #1
                broadcast_index(out_idx, out_shape, in_shape, in_idx)           |
                                                                                |
                out_pos = index_to_position(out_idx, out_strides)               |
                in_pos = index_to_position(in_idx, in_strides)                  |
                                                                                |
                out[out_pos] = fn(in_storage[in_pos])                           |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (210) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (207) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (244)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (244)
---------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                          |
        out: Storage,                                                                  |
        out_shape: Shape,                                                              |
        out_strides: Strides,                                                          |
        a_storage: Storage,                                                            |
        a_shape: Shape,                                                                |
        a_strides: Strides,                                                            |
        b_storage: Storage,                                                            |
        b_shape: Shape,                                                                |
        b_strides: Strides,                                                            |
    ) -> None:                                                                         |
        # When tensors are aligned                                                     |
        if check_arrays_match(                                                         |
            out_strides, a_strides, out_shape, a_shape                                 |
        ) and check_arrays_match(out_strides, b_strides, out_shape, b_shape):          |
            for i in prange(len(out)):-------------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                                |
        else:                                                                          |
            # When tensors are not aligned                                             |
            for i in prange(len(out)):-------------------------------------------------| #8
                out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------------------| #4
                a_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)----------------------| #5
                b_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)----------------------| #6
                                                                                       |
                to_index(i, out_shape, out_idx)                                        |
                broadcast_index(out_idx, out_shape, a_shape, a_idx)                    |
                broadcast_index(out_idx, out_shape, b_shape, b_idx)                    |
                                                                                       |
                out_pos = index_to_position(out_idx, out_strides)                      |
                a_pos = index_to_position(a_idx, a_strides)                            |
                b_pos = index_to_position(b_idx, b_strides)                            |
                                                                                       |
                out[out_pos] = fn(float(a_storage[a_pos]), float(b_storage[b_pos]))    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (264) is
hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (265) is
hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (266) is
hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (302)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (302)
------------------------------------------------------------------|loop #ID
    def _reduce(                                                  |
        out: Storage,                                             |
        out_shape: Shape,                                         |
        out_strides: Strides,                                     |
        a_storage: Storage,                                       |
        a_shape: Shape,                                           |
        a_strides: Strides,                                       |
        reduce_dim: int,                                          |
    ) -> None:                                                    |
        # Perform reduction in parallel                           |
        for i in prange(len(out)):--------------------------------| #10
            out_idx = np.zeros(len(out_shape), dtype=np.int32)----| #9
            to_index(i, out_shape, out_idx)                       |
            out_pos = index_to_position(out_idx, out_strides)     |
            reduce_size = a_shape[reduce_dim]                     |
            for s in range(reduce_size):                          |
                out_idx[reduce_dim] = s                           |
                a_pos = index_to_position(out_idx, a_strides)     |
                out[i] = fn(out[out_pos], a_storage[a_pos])       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (313) is
hoisted out of the parallel loop labelled #10 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (325)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/angelagui/Documents/MLE/mod3-xgui2001/minitorch/fast_ops.py (325)
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                |
    out: Storage,                                                                           |
    out_shape: Shape,                                                                       |
    out_strides: Strides,                                                                   |
    a_storage: Storage,                                                                     |
    a_shape: Shape,                                                                         |
    a_strides: Strides,                                                                     |
    b_storage: Storage,                                                                     |
    b_shape: Shape,                                                                         |
    b_strides: Strides,                                                                     |
) -> None:                                                                                  |
    """NUMBA tensor matrix multiply function.                                               |
                                                                                            |
    Should work for any tensor shapes that broadcast as long as                             |
                                                                                            |
    ```                                                                                     |
    assert a_shape[-1] == b_shape[-2]                                                       |
    ```                                                                                     |
                                                                                            |
    Optimizations:                                                                          |
                                                                                            |
    * Outer loop in parallel                                                                |
    * No index buffers or function calls                                                    |
    * Inner loop should have no global writes, 1 multiply.                                  |
                                                                                            |
                                                                                            |
    Args:                                                                                   |
    ----                                                                                    |
        out (Storage): storage for `out` tensor                                             |
        out_shape (Shape): shape for `out` tensor                                           |
        out_strides (Strides): strides for `out` tensor                                     |
        a_storage (Storage): storage for `a` tensor                                         |
        a_shape (Shape): shape for `a` tensor                                               |
        a_strides (Strides): strides for `a` tensor                                         |
        b_storage (Storage): storage for `b` tensor                                         |
        b_shape (Shape): shape for `b` tensor                                               |
        b_strides (Strides): strides for `b` tensor                                         |
                                                                                            |
    Returns:                                                                                |
    -------                                                                                 |
        None : Fills in `out`                                                               |
                                                                                            |
    """                                                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  |
                                                                                            |
    assert a_shape[-1] == b_shape[-2]                                                       |
                                                                                            |
    batch_size = out_shape[0]  # Number of matrices in batch                                |
    rows = a_shape[-2]  # Number of rows                                                    |
    cols = b_shape[-1]  # Number of columns                                                 |
    reduce_dim = a_shape[-1]  # Dimension to sum over                                       |
                                                                                            |
    # Matrix multiplication loop                                                            |
    for batch in prange(batch_size):--------------------------------------------------------| #13
        for row in prange(rows):------------------------------------------------------------| #12
            for col in prange(cols):--------------------------------------------------------| #11
                # Calculate position in output tensor                                       |
                out_idx = (                                                                 |
                    batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]    |
                )                                                                           |
                                                                                            |
                # Dot product computation                                                   |
                result = 0.0                                                                |
                for k in range(reduce_dim):                                                 |
                    # Get positions in input tensors                                        |
                    a_idx = (                                                               |
                        batch * a_batch_stride + row * a_strides[1] + k * a_strides[2]      |
                    )                                                                       |
                    b_idx = (                                                               |
                        batch * b_batch_stride + k * b_strides[1] + col * b_strides[2]      |
                    )                                                                       |
                                                                                            |
                    result += a_storage[a_idx] * b_storage[b_idx]                           |
                                                                                            |
                out[out_idx] = result                                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)



Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

* 3.4 Timing Optimizations

| Size  | Fast (CPU) | GPU (Optimized) |
|-------|------------|-----------------|
| 64    | 0.00318    | 0.00568         |
| 128   | 0.01637    | 0.01250         |
| 256   | 0.09008    | 0.04788         |
| 512   | 0.96055    | 0.19299         |
| 1024  | 7.46985    | 0.97894         |

![Graph](./output.png)