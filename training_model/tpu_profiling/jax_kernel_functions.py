"""Common JAX kernel implementations and a simple registration decorator.

This module defines a lightweight registry for JAX-based kernel functions and a
`register_kernel` decorator to add functions to that registry by name. It also
includes a couple of example kernels used for profiling and reference tests.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp

########################################################
# JAX Kernel Function Registry
########################################################
# Global registry mapping kernel name to callable
KERNEL_REGISTRY: Dict[str, Callable] = {}
CONSTANT_PROFILING_KERNEL_NAME = "realops_profiling_kernel"


def register_kernel(name: str):
    """Decorator to register a function under a kernel name.

    Example:
        @register_kernel("relu")
        def relu(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.maximum(x, 0)
    """

    def decorator(func):
        # Registration occurs at import time when the function is defined
        KERNEL_REGISTRY[name] = func
        return func

    return decorator


########################################################
# JAX Kernel Functions
########################################################
@register_kernel("matrix_multiply")
def profiling_matrix_multiply(
    input_a: jnp.ndarray,
    input_b: jnp.ndarray,
    parameters: Optional[Dict[str, Any]] = None,
) -> jnp.ndarray:
    """Matrix multiplication kernel.

    Args:
        input_a: Left-hand matrix.
        input_b: Right-hand matrix.
        parameters: Optional configuration; unused by this simple kernel.

    Returns:
        The matrix product of ``input_a`` and ``input_b``.
    """
    return jnp.matmul(input_a, input_b)

@register_kernel("relu")
def profiling_relu(input_a: jnp.ndarray, parameters: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
    """ReLU activation kernel: elementwise max(x, 0)."""
    return jnp.maximum(input_a, 0)


@register_kernel("add")
def profiling_add(input_a: jnp.ndarray, input_b: jnp.ndarray, parameters: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
    """Addition kernel."""
    return jnp.add(input_a, input_b)

@register_kernel("mul")
def profiling_mul(input_a: jnp.ndarray, input_b: jnp.ndarray, parameters: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
    """Multiplication kernel."""
    return jnp.multiply(input_a, input_b)

########################################################
# JAX Kernel Wrapper Class
########################################################
class KernelWarpper:
    def __init__(self, kernel_name: str, input_structs: List[Tuple[Tuple[int, ...], jnp.dtype]], parameters: Optional[Dict[str, Any]] = {}):
        self.kernel_name = kernel_name
        self.input_structs = input_structs
        self.parameters = parameters

        self.callable_function = KERNEL_REGISTRY[self.kernel_name]
        self.jit_lower = None
        self.jit_compiled_function = None

    def compile(self):
        jax_input_structs = []
        for shape, dtype in self.input_structs:
            jax_input_structs.append(jax.ShapeDtypeStruct(shape, dtype))

        # NOTE: Do not change the name of the function, it is used for profiling
        def compiled_kernel_function(*jax_array_inputs):
            return self.callable_function(*jax_array_inputs, parameters=self.parameters)


        self.jit_lower = jax.jit(jax.named_call(compiled_kernel_function, name=self.kernel_name)).lower(*jax_input_structs)
        self.jit_compiled_function = self.jit_lower.compile()

    def get_compiled_function(self) -> Callable[..., jnp.ndarray]:
        assert self.jit_compiled_function is not None, "Kernel not compiled"
        return self.jit_compiled_function

    def get_input_structs(self):
        return self.input_structs

    def get_kernel_name(self) -> str:
        return self.kernel_name