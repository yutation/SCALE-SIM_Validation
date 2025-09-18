import jax
import jax.numpy as jnp
from enum import Enum
from typing import Callable, Tuple

# Kernel functions
def validation_matrix_multiply(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(input_A, input_B)

def validation_dot_product(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(input_A, input_B)

def validation_convolve(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.convolve(input_A, input_B)

def validation_convolve2d(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jax.scipy.signal.convolve2d(input_A, input_B)

def validation_convolve_scalesim(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.conv_general_dilated(input_A, input_B, (1, 1), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW"))

def validation_vector_add(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.add(input_A, input_B)

def validation_vector_sub(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.subtract(input_A, input_B)

def validation_vector_mul(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.multiply(input_A, input_B)

def validation_vector_div(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.divide(input_A, input_B)

def validation_vector_and(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.bitwise_and(input_A, input_B)

def validation_vector_or(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.bitwise_or(input_A, input_B)

def validation_vector_shl(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.left_shift(input_A, input_B)

def validation_vector_shr(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jnp.right_shift(input_A, input_B)

def validation_relu(input_A: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(input_A, 0)

def validation_sigmoid(input_A: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation function: 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + jnp.exp(-input_A))

def validation_tanh(input_A: jnp.ndarray) -> jnp.ndarray:
    """Hyperbolic tangent activation function"""
    return jnp.tanh(input_A)

def validation_leaky_relu(input_A: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """Leaky ReLU activation function: max(alpha * x, x)"""
    return jnp.maximum(alpha * input_A, input_A)

def validation_elu(input_A: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    """Exponential Linear Unit activation function"""
    return jnp.where(input_A > 0, input_A, alpha * (jnp.exp(input_A) - 1))

def validation_selu(input_A: jnp.ndarray, alpha: float = 1.6732632423543772848170429916717, 
                    scale: float = 1.0507009873554804934193349852946) -> jnp.ndarray:
    """Scaled Exponential Linear Unit activation function"""
    return scale * jnp.where(input_A > 0, input_A, alpha * (jnp.exp(input_A) - 1))

def validation_parametric_relu(input_A: jnp.ndarray, alpha: float = 0.25) -> jnp.ndarray:
    """Parametric ReLU activation function with learnable parameter alpha"""
    return jnp.where(input_A > 0, input_A, alpha * input_A)

def validation_binary_step(input_A: jnp.ndarray, threshold: float = 0.0) -> jnp.ndarray:
    """Binary step function: 1 if x > threshold, 0 otherwise"""
    return jnp.where(input_A > threshold, 1.0, 0.0)

def validation_linear(input_A: jnp.ndarray, slope: float = 1.0, bias: float = 0.0) -> jnp.ndarray:
    """Linear activation function: slope * x + bias"""
    return slope * input_A + bias

def validation_batch_norm(input_A: jnp.ndarray, gamma: jnp.ndarray = None, beta: jnp.ndarray = None, 
                         running_mean: jnp.ndarray = None, running_var: jnp.ndarray = None,
                         training: bool = False, momentum: float = 0.1, eps: float = 1e-5, 
                         axis: int = -1) -> tuple:
    """
    Batch Normalization: normalizes input across batch dimension
    
    Args:
        input_A: Input tensor
        gamma: Scale parameter (default: ones)
        beta: Shift parameter (default: zeros)
        running_mean: Running mean for inference (default: zeros)
        running_var: Running variance for inference (default: ones)
        training: Whether in training mode (True) or inference mode (False)
        momentum: Momentum for running statistics update (default: 0.1)
        eps: Small constant for numerical stability
        axis: Axis along which to normalize (default: -1, last axis)
        
    Returns:
        tuple: (normalized_output, updated_running_mean, updated_running_var)
               During inference, running statistics are not updated
    """
    # Initialize parameters if not provided
    feature_size = input_A.shape[axis]
    
    if gamma is None:
        gamma = jnp.ones(feature_size)
    if beta is None:
        beta = jnp.zeros(feature_size)
    if running_mean is None:
        running_mean = jnp.zeros(feature_size)
    if running_var is None:
        running_var = jnp.ones(feature_size)
    
    # Calculate axes to reduce over (all except the feature axis)
    reduce_axes = tuple(i for i in range(input_A.ndim) if i != axis)
    
    if training:
        # Training mode: use batch statistics
        batch_mean = jnp.mean(input_A, axis=reduce_axes, keepdims=True)
        batch_var = jnp.var(input_A, axis=reduce_axes, keepdims=True)
        
        # Use batch statistics for normalization
        mean_for_norm = batch_mean
        var_for_norm = batch_var
        
        # Update running statistics
        # Remove keepdims for running statistics update
        batch_mean_scalar = jnp.squeeze(batch_mean, axis=reduce_axes)
        batch_var_scalar = jnp.squeeze(batch_var, axis=reduce_axes)
        
        new_running_mean = (1 - momentum) * running_mean + momentum * batch_mean_scalar
        new_running_var = (1 - momentum) * running_var + momentum * batch_var_scalar
        
    else:
        # Inference mode: use running statistics
        # Reshape running statistics to match input dimensions for broadcasting
        shape_for_broadcast = [1] * input_A.ndim
        shape_for_broadcast[axis] = feature_size
        
        mean_for_norm = running_mean.reshape(shape_for_broadcast)
        var_for_norm = running_var.reshape(shape_for_broadcast)
        
        # Don't update running statistics during inference
        new_running_mean = running_mean
        new_running_var = running_var
    
    # Normalize
    normalized = (input_A - mean_for_norm) / jnp.sqrt(var_for_norm + eps)
    
    # Apply affine transformation
    # Broadcast gamma and beta to match input shape
    shape = [1] * input_A.ndim
    shape[axis] = feature_size
    gamma = gamma.reshape(shape)
    beta = beta.reshape(shape)
    
    output = gamma * normalized + beta
    
    return output, new_running_mean, new_running_var

def validation_batch_norm_simple_training(input_A: jnp.ndarray, axis: int = -1, 
                                eps: float = 1e-5) -> jnp.ndarray:
    """
    Simplified Batch Normalization that computes normalization over specified axis.
    
    Args:
        input_A: Input tensor
        axis: Axis to normalize over (default: -1)
        eps: Small constant for numerical stability
    """
    # Create gamma and beta with reduced shape - JAX will handle broadcasting automatically
    reduced_shape = input_A.shape[axis]
    gamma = jnp.ones(reduced_shape)
    beta = jnp.zeros(reduced_shape)
    
    # Calculate axes to reduce over (all except the specified axis)
    all_axes = set(range(input_A.ndim))
    keep_axes = {axis}
    reduce_axes = tuple(all_axes - keep_axes)
    
    mean = jnp.mean(input_A, axis=reduce_axes, keepdims=True)
    var = jnp.var(input_A, axis=reduce_axes, keepdims=True)
    
    normalized = (input_A - mean) / jnp.sqrt(var + eps)
    return gamma * normalized + beta

def validation_batch_norm_simple_inference(input_A: jnp.ndarray, axis: int = -1, 
                                eps: float = 1e-5) -> jnp.ndarray:
    """
    Simplified Batch Normalization that computes normalization over specified axis.
    
    Args:
        input_A: Input tensor
        axis: Axis to normalize over (default: -1)
        eps: Small constant for numerical stability
    """
    mean = jnp.zeros(input_A.shape[axis])
    var = jnp.ones(input_A.shape[axis])
    gamma = jnp.ones(input_A.shape[axis])
    beta = jnp.zeros(input_A.shape[axis])

    normalized = (input_A - mean) / jnp.sqrt(var + eps)
    return gamma * normalized + beta    


def validation_layer_norm(input_A: jnp.ndarray, gamma: jnp.ndarray = None, beta: jnp.ndarray = None, 
                         eps: float = 1e-5, axis: int = -1) -> jnp.ndarray:
    """
    Layer Normalization: normalizes input across feature dimension
    Args:
        input_A: Input tensor
        gamma: Scale parameter (default: ones)
        beta: Shift parameter (default: zeros)
        eps: Small constant for numerical stability
        axis: Axis along which to normalize (default: -1, last axis)
    """
    if gamma is None:
        gamma = jnp.ones(input_A.shape[axis])
    if beta is None:
        beta = jnp.zeros(input_A.shape[axis])
    
    # Calculate mean and variance along the specified axis
    mean = jnp.mean(input_A, axis=axis, keepdims=True)
    var = jnp.var(input_A, axis=axis, keepdims=True)
    
    # Normalize and apply affine transformation
    normalized = (input_A - mean) / jnp.sqrt(var + eps)
    
    # Broadcast gamma and beta to match input shape
    shape = [1] * input_A.ndim
    shape[axis] = input_A.shape[axis]
    gamma = gamma.reshape(shape)
    beta = beta.reshape(shape)
    
    return gamma * normalized + beta

def validation_layer_norm_simple(input_A: jnp.ndarray, axis: Tuple[int, ...] = (-1,), 
                              eps: float = 1e-5) -> jnp.ndarray:
    """
    Simplified Layer Normalization that computes normalization over multiple axes.
    
    Args:
        input_A: Input tensor
        axis: Tuple of axes to normalize over (default: (-1,))
        eps: Small constant for numerical stability
    """
    # Create gamma and beta with reduced shape - JAX will handle broadcasting automatically
    reduced_shape = tuple(input_A.shape[ax] for ax in axis)
    gamma = jnp.ones(reduced_shape)
    beta = jnp.zeros(reduced_shape)
    mean = jnp.mean(input_A, axis=axis, keepdims=True)
    var = jnp.var(input_A, axis=axis, keepdims=True)
    normalized = (input_A - mean) / jnp.sqrt(var + eps)
    return gamma * normalized + beta


def validation_rms_norm(input_A: jnp.ndarray, gamma: jnp.ndarray = None, eps: float = 1e-5, 
                       axis: int = -1) -> jnp.ndarray:
    """
    RMS (Root Mean Square) Normalization: normalizes by RMS value
    Args:
        input_A: Input tensor
        gamma: Scale parameter (default: ones)
        eps: Small constant for numerical stability
        axis: Axis along which to normalize (default: -1, last axis)
    """
    if gamma is None:
        gamma = jnp.ones(input_A.shape[axis])
    
    # Calculate RMS (root mean square)
    rms = jnp.sqrt(jnp.mean(jnp.square(input_A), axis=axis, keepdims=True) + eps)
    
    # Normalize by RMS
    normalized = input_A / rms
    
    # Broadcast gamma to match input shape
    shape = [1] * input_A.ndim
    shape[axis] = input_A.shape[axis]
    gamma = gamma.reshape(shape)
    
    return gamma * normalized

def validation_instance_norm(input_A: jnp.ndarray, gamma: jnp.ndarray = None, beta: jnp.ndarray = None, 
                            eps: float = 1e-5) -> jnp.ndarray:
    """
    Instance Normalization: normalizes each sample and channel independently
    Typically used for style transfer and GANs. For a 4D tensor (N, C, H, W),
    normalization is applied over the spatial dimensions (H, W) for each sample and channel.
    
    Args:
        input_A: Input tensor, typically shape (N, C, H, W) for images
        gamma: Scale parameter (default: ones), shape should match channel dimension
        beta: Shift parameter (default: zeros), shape should match channel dimension  
        eps: Small constant for numerical stability
    """
    if input_A.ndim < 2:
        raise ValueError("Instance normalization requires at least 2D input")
    
    # For typical use case (N, C, H, W), normalize over spatial dimensions (H, W)
    # For each sample and channel independently
    if input_A.ndim == 4:  # (N, C, H, W)
        # Normalize over spatial dimensions (H, W) for each (N, C)
        reduce_axes = (2, 3)
        param_shape = (1, input_A.shape[1], 1, 1)  # Shape for broadcasting
        param_size = input_A.shape[1]  # Channel dimension
    elif input_A.ndim == 3:  # (N, C, L) - e.g., 1D sequences
        # Normalize over length dimension for each (N, C)
        reduce_axes = (2,)
        param_shape = (1, input_A.shape[1], 1)
        param_size = input_A.shape[1]
    elif input_A.ndim == 2:  # (N, C)
        # Normalize over channel dimension for each sample
        reduce_axes = (1,)
        param_shape = (1, input_A.shape[1])
        param_size = input_A.shape[1]
    else:
        # General case: normalize over all dimensions except the first two (N, C, ...)
        reduce_axes = tuple(range(2, input_A.ndim))
        param_shape = [1] * input_A.ndim
        param_shape[1] = input_A.shape[1]  # Keep channel dimension
        param_shape = tuple(param_shape)
        param_size = input_A.shape[1]
    
    if gamma is None:
        gamma = jnp.ones(param_size)
    if beta is None:
        beta = jnp.zeros(param_size)
    
    # Calculate mean and variance over spatial/feature dimensions
    mean = jnp.mean(input_A, axis=reduce_axes, keepdims=True)
    var = jnp.var(input_A, axis=reduce_axes, keepdims=True)
    
    # Normalize
    normalized = (input_A - mean) / jnp.sqrt(var + eps)
    
    # Reshape gamma and beta for broadcasting
    gamma = gamma.reshape(param_shape)
    beta = beta.reshape(param_shape)
    
    return gamma * normalized + beta



class ScaleSimTopologyType(Enum):
    GEMM = "gemm"
    CONV = "conv"


class KernelType(Enum):
    MATRIX_MULTIPLY = "matrix_multiply"
    DOT_PRODUCT = "dot_product"
    CONVOLVE = "convolve"
    CONVOLVE2D = "convolve2d"
    CONVOLVE_SCALESIM = "convolve_scalesim"
    VECTOR_ADD = "vector_add"
    VECTOR_SUB = "vector_sub"
    VECTOR_MUL = "vector_mul"
    VECTOR_DIV = "vector_div"
    VECTOR_AND = "vector_and"
    VECTOR_OR = "vector_or"
    VECTOR_SHL = "vector_shl"
    VECTOR_SHR = "vector_shr"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    PARAMETRIC_RELU = "parametric_relu"
    BINARY_STEP = "binary_step"
    LINEAR = "linear"
    BATCH_NORM = "batch_norm"
    BATCH_NORM_SIMPLE_TRAINING = "batch_norm_simple_training"
    BATCH_NORM_SIMPLE_INFERENCE = "batch_norm_simple_inference"
    LAYER_NORM = "layer_norm"
    LAYER_NORM_SIMPLE = "layer_norm_simple"
    RMS_NORM = "rms_norm"
    INSTANCE_NORM = "instance_norm"
    


    def get_kernel(self) -> Callable:
        if self == KernelType.MATRIX_MULTIPLY:
            return validation_matrix_multiply
        elif self == KernelType.DOT_PRODUCT:
            return validation_dot_product
        elif self == KernelType.CONVOLVE:
            return validation_convolve
        elif self == KernelType.CONVOLVE2D:
            return validation_convolve2d
        elif self == KernelType.CONVOLVE_SCALESIM:
            return validation_convolve_scalesim
        elif self == KernelType.VECTOR_ADD:
            return validation_vector_add
        elif self == KernelType.VECTOR_SUB:
            return validation_vector_sub
        elif self == KernelType.VECTOR_MUL:
            return validation_vector_mul
        elif self == KernelType.VECTOR_DIV:
            return validation_vector_div
        elif self == KernelType.VECTOR_AND:
            return validation_vector_and
        elif self == KernelType.VECTOR_OR:
            return validation_vector_or
        elif self == KernelType.VECTOR_SHL:
            return validation_vector_shl
        elif self == KernelType.VECTOR_SHR:
            return validation_vector_shr
        elif self == KernelType.RELU:
            return validation_relu
        elif self == KernelType.SIGMOID:
            return validation_sigmoid
        elif self == KernelType.TANH:
            return validation_tanh
        elif self == KernelType.LEAKY_RELU:
            return validation_leaky_relu
        elif self == KernelType.ELU:
            return validation_elu
        elif self == KernelType.SELU:
            return validation_selu
        elif self == KernelType.PARAMETRIC_RELU:
            return validation_parametric_relu
        elif self == KernelType.BINARY_STEP:
            return validation_binary_step
        elif self == KernelType.LINEAR:
            return validation_linear
        elif self == KernelType.BATCH_NORM:
            return validation_batch_norm
        elif self == KernelType.BATCH_NORM_SIMPLE_TRAINING:
            return validation_batch_norm_simple_training
        elif self == KernelType.BATCH_NORM_SIMPLE_INFERENCE:
            return validation_batch_norm_simple_inference
        elif self == KernelType.LAYER_NORM:
            return validation_layer_norm
        elif self == KernelType.LAYER_NORM_SIMPLE:
            return validation_layer_norm_simple
        elif self == KernelType.RMS_NORM:
            return validation_rms_norm
        elif self == KernelType.INSTANCE_NORM:
            return validation_instance_norm
        else:
            raise ValueError(f"Unknown kernel type: {self}")
    
    def get_scale_sim_topology_type(self) -> ScaleSimTopologyType:
        if self == KernelType.MATRIX_MULTIPLY or self == KernelType.DOT_PRODUCT:
            return ScaleSimTopologyType.GEMM
        elif self == KernelType.CONVOLVE or self == KernelType.CONVOLVE2D or self == KernelType.CONVOLVE_SCALESIM:
            return ScaleSimTopologyType.CONV
        else:
            raise ValueError(f"Unknown kernel type: {self}")