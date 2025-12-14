#!/usr/bin/env python3
"""Test the trace_calls decorator."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging to DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

from logger import trace_calls


@trace_calls()
def simple_function(x, y):
    """A simple function to test."""
    return x + y


@trace_calls(show_return=True)
def function_with_return(a, b, c=10):
    """Function with return value shown."""
    result = a * b + c
    return result


@trace_calls()
def nested_function(value):
    """Function that calls another traced function."""
    result1 = simple_function(value, 5)
    result2 = function_with_return(result1, 2)
    return result2


if __name__ == "__main__":
    print("Testing trace_calls decorator...\n")

    print("1. Simple function:")
    result = simple_function(3, 4)
    print(f"Result: {result}\n")

    print("2. Function with return value:")
    result = function_with_return(5, 6, c=20)
    print(f"Result: {result}\n")

    print("3. Nested function calls:")
    result = nested_function(10)
    print(f"Result: {result}\n")

    print("Done!")
