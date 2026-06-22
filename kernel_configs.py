from typing import Tuple
import flexible_validation as fv
import jax.numpy as jnp
import kernel_functions as kf

def generate_matrix_multiply_config(name: str, M, N, K) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.MATRIX_MULTIPLY,
        inputs=[((M, K), jnp.float16),
                ((K, N), jnp.float16)]
    )

# Example function


def generate_dot_product_config(name: str, mnk_value: int) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.DOT_PRODUCT,
        inputs=[((mnk_value, ), jnp.float16),
                ((mnk_value, ), jnp.float16)]
    )

def generate_convolve2d_config(name: str, mnk_value: int) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.CONVOLVE2D,
        inputs=[((mnk_value, mnk_value), jnp.float16),
                ((3, 3), jnp.float16),]
    )

def generate_conv_nchw_config(name: str, N: int, C: int, H: int, W: int, K: int, R: int, S: int) -> fv.ValidationConfig:
    """Generate convolution config with NCHW input format and OIHW filter format."""
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.CONVOLVE_SCALESIM,
        inputs=[((N, C, H, W), jnp.float16),
                ((K, C, R, S), jnp.float16)]
    )


def generate_vector_op_config(name: str, kernel_type: kf.KernelType, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kernel_type,
        inputs=[(shape, jnp.float16),
                (shape, jnp.float16)]
    )

def generate_activation_config(name: str, kernel_type: kf.KernelType, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kernel_type,
        inputs=[(shape, jnp.float16)]
    )


def generate_layer_norm_config(name: str, shape: Tuple[int, ...], axis: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.LAYER_NORM_SIMPLE,
        inputs=[(shape, jnp.float16)],  
        kernel_params={"axis": axis}
    )

def generate_rms_norm_config(name: str, shape: Tuple[int, ...], axis: int) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.RMS_NORM_SIMPLE,
        inputs=[(shape, jnp.float16)],
        kernel_params={"axis": axis}
    )

def generate_batch_norm_training_config(name: str, shape: Tuple[int, ...], axis: int) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.BATCH_NORM_SIMPLE_TRAINING,
        inputs=[(shape, jnp.float16)],
        kernel_params={"axis": axis}
    )

def generate_batch_norm_inference_config(name: str, shape: Tuple[int, ...], axis: int) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.BATCH_NORM_SIMPLE_INFERENCE,
        inputs=[(shape, jnp.float16)],
        kernel_params={"axis": axis}
    )

def generate_max_pooling_config(name: str, shape: Tuple[int, ...], window_shape: Tuple[int, ...] = (2, 2), 
                               strides: Tuple[int, ...] = (2, 2), padding: str = "VALID") -> fv.ValidationConfig:
    """Generate max pooling config for NCHW input format."""
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.MAX_POOLING,
        inputs=[(shape, jnp.float16)],
        kernel_params={"window_shape": window_shape, "strides": strides, "padding": padding}
    )

def generate_avg_pooling_config(name: str, shape: Tuple[int, ...], window_shape: Tuple[int, ...] = (2, 2), 
                               strides: Tuple[int, ...] = (2, 2), padding: str = "VALID") -> fv.ValidationConfig:
    """Generate average pooling config for NCHW input format."""
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.AVG_POOLING,
        inputs=[(shape, jnp.float16)],
        kernel_params={"window_shape": window_shape, "strides": strides, "padding": padding}
    )

def generate_broadcast_to_dim_config(name: str, input_shape: Tuple[int, ...], shape: Tuple[int, ...], broadcast_dimensions: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.BROADCAST_TO_DIM,
        inputs=[(input_shape, jnp.float16)],
        kernel_params={"shape": shape, "broadcast_dimensions": broadcast_dimensions}
    )

def generate_product_reduce_config(name: str, shape: Tuple[int, ...], axis: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.PRODUCT_REDUCE,
        inputs=[(shape, jnp.float16)],
        kernel_params={"axis": axis}
    )

def generate_sum_reduce_config(name: str, shape: Tuple[int, ...], axis: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.SUM_REDUCE,
        inputs=[(shape, jnp.float16)],
        kernel_params={"axis": axis}
    )

def generate_reshape_config(name: str, shape: Tuple[int, ...], newshape: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.RESHAPE,
        inputs=[(shape, jnp.float16)],
        kernel_params={"newshape": newshape}
    )

def generate_transpose_config(name: str, shape: Tuple[int, ...], perm: Tuple[int, ...]) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=kf.KernelType.TRANSPOSE,
        inputs=[(shape, jnp.float16)],
        kernel_params={"perm": perm}
    )