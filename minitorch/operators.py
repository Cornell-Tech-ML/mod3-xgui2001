"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiply two floats."""
    return a * b


def id(a: float) -> float:
    """Return the identity of a float."""
    return a


def add(a: float, b: float) -> float:
    """Add two floats."""
    return a + b


def neg(a: float) -> float:
    """Negate a float."""
    return -a


def lt(a: float, b: float) -> bool:
    """Compare two floats."""
    return a < b


def eq(a: float, b: float) -> bool:
    """Compare two floats."""
    return a == b


def max(a: float, b: float) -> float:
    """Return the maximum of two floats."""
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Check if two floats are close."""
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid of a float."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(a: float) -> float:
    """Compute the relu of a float."""
    return a if a > 0 else 0.0


def log(a: float) -> float:
    """Compute the log of a float."""
    return math.log(a)


def exp(a: float) -> float:
    """Compute the exponential of a float."""
    return math.exp(a)


def log_back(a: float, dout: float) -> float:
    """Compute the derivative of the log function."""
    return dout / a


def inv(a: float) -> float:
    """Compute the inverse of a float."""
    return 1.0 / a


def inv_back(a: float, dout: float) -> float:
    """Compute the derivative of the inverse function."""
    return -dout / (a * a)


def relu_back(a: float, dout: float) -> float:
    """Compute the derivative of the relu function."""
    return dout * (a > 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce


def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list.

    Args:
    ----
        fn: The function to apply to each element.
        lst: The input iterable of floats.

    Returns:
    -------
        An iterable of floats with the function applied to each element.

    """
    return [fn(x) for x in lst]


def zipWith(
    fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> Iterable[float]:
    """Combines elements from two iterables using a given function.

    Args:
    ----
        fn (Callable[[float, float], float]): Function to apply to pairs of elements.
        lst1 (Iterable[float]): First iterable of floats.
        lst2 (Iterable[float]): Second iterable of floats.

    Returns:
    -------
        Iterable[float]: Result of applying fn to pairs of elements from lst1 and lst2.

    """
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(
    fn: Callable[[float, float], float], lst: Iterable[float], initial: float
) -> float:
    """Reduces an iterable to a single value using a given function.

    Args:
    ----
        fn (Callable[[float, float], float]): Function to apply cumulatively to the elements.
        lst (Iterable[float]): Iterable of floats to reduce.
        initial (float): Initial value for the reduction.

    Returns:
    -------
        float: Result of applying fn cumulatively to the elements of lst.

    """
    result = initial
    for x in lst:
        result = fn(result, x)
    return result


# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate each element in the given list.

    Args:
    ----
        lst (Iterable[float]): The input list of floats.

    Returns:
    -------
        Iterable[float]: A list containing the negated values of the input.

    """
    return map(neg, lst)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists together.

    Args:
    ----
        ls1 (Iterable[float]): The first input list of floats.
        ls2 (Iterable[float]): The second input list of floats.

    Returns:
    -------
        Iterable[float]: A list containing the sum of the input lists.

    """
    return zipWith(add, ls1, ls2)


def sum(lst: Iterable[float]) -> float:
    """Calculate the sum of all elements in the given list.

    Args:
    ----
        lst (Iterable[float]): The input list of floats.

    Returns:
    -------
        float: The sum of all elements in the input list.

    """
    return reduce(add, lst, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in the given list.

    Args:
    ----
        ls: The input iterable of floats.

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(mul, ls, 1.0)
