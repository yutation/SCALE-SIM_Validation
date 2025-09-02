from typing import Tuple
import flexible_validation as fv
import jax.numpy as jnp

def generate_matrix_multiply_config(name: str, M, N, K) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.MATRIX_MULTIPLY,
        inputs=[((M, K), jnp.float16),
                ((K, N), jnp.float16)]
    )

# Example function


def generate_dot_product_config(name: str, mnk_value: int) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.DOT_PRODUCT,
        inputs=[((mnk_value, ), jnp.float16),
                ((mnk_value, ), jnp.float16)]
    )

def generate_convolve2d_config(name: str, mnk_value: int) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.CONVOLVE2D,
        inputs=[((mnk_value, mnk_value), jnp.float16),
                ((3, 3), jnp.float16),]
    )

def generate_conv_nchw_config(name: str, N: int, C: int, H: int, W: int, K: int, R: int, S: int) -> fv.ValidationConfig:
    """Generate convolution config with NCHW input format and OIHW filter format."""
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.CONVOLVE_SCALESIM,
        inputs=[((N, C, H, W), jnp.float16),
                ((K, C, R, S), jnp.float16)]
    )


def generate_vector_add_config(name: str, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.VECTOR_ADD,
        inputs=[(shape, jnp.float16),
                (shape, jnp.float16)]
    )

def generate_vector_sub_config(name: str, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.VECTOR_SUB,
        inputs=[(shape, jnp.float16),
                (shape, jnp.float16)]
    )

def generate_vector_mul_config(name: str, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.VECTOR_MUL,
        inputs=[(shape, jnp.float16),
                (shape, jnp.float16)]
    )

def generate_vector_div_config(name: str, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.VECTOR_DIV,
        inputs=[(shape, jnp.float16),
                (shape, jnp.float16)]
    )

def generate_relu_config(name: str, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.RELU,
        inputs=[(shape, jnp.float16)]
    )