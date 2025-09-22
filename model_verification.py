import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc
import kernel_functions as kf
import operation_classification as oc
from typing import Dict, Callable, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from latency_prediction import PredictionManager


# Mapping between operation types and their corresponding kernel config generators
OPERATION_TO_KERNEL_CONFIG_MAP: Dict[oc.OperationBase, Callable] = {
    # Elementwise operations
    oc.OperationElementwise.ADD: lambda name, shape: kc.generate_vector_op_config(name, kf.KernelType.VECTOR_ADD, shape),
    oc.OperationElementwise.SUBTRACT: lambda name, shape: kc.generate_vector_op_config(name, kf.KernelType.VECTOR_SUB, shape),
    oc.OperationElementwise.MULTIPLY: lambda name, shape: kc.generate_vector_op_config(name, kf.KernelType.VECTOR_MUL, shape),
    oc.OperationElementwise.DIVIDE: lambda name, shape: kc.generate_vector_op_config(name, kf.KernelType.VECTOR_DIV, shape),
    
    # Activation operations
    oc.OperationActivation.RELU: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.RELU, shape),
    oc.OperationActivation.SIGMOID: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.SIGMOID, shape),
    oc.OperationActivation.TANH: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.TANH, shape),
    oc.OperationActivation.LEAKY_RELU: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.LEAKY_RELU, shape),
    oc.OperationActivation.ELU: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.ELU, shape),
    oc.OperationActivation.SELU: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.SELU, shape),
    oc.OperationActivation.PARAMETRIC_RELU: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.PARAMETRIC_RELU, shape),
    oc.OperationActivation.LINEAR: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.LINEAR, shape),
    
    # Normalization operations
    oc.OperationNormalization.BATCH_NORM: lambda name, shape, axis=-1: kc.generate_batch_norm_config(name, shape, axis),
    oc.OperationNormalization.LAYER_NORM: lambda name, shape, axis=(-1,): kc.generate_layer_norm_config(name, shape, axis),
    oc.OperationNormalization.RMS_NORM: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.RMS_NORM, shape),
    oc.OperationNormalization.INSTANCE_NORM: lambda name, shape: kc.generate_activation_config(name, kf.KernelType.INSTANCE_NORM, shape),
    
    # Matrix multiplication operations
    oc.OperationMatmul.LINEAR: lambda name, M, N, K: kc.generate_matrix_multiply_config(name, M, N, K),
}

# Direct mapping between operation types and kernel types for simple cases
OPERATION_TO_KERNEL_TYPE_MAP: Dict[oc.OperationBase, kf.KernelType] = {
    # Elementwise operations
    oc.OperationElementwise.ADD: kf.KernelType.VECTOR_ADD,
    oc.OperationElementwise.SUBTRACT: kf.KernelType.VECTOR_SUB,
    oc.OperationElementwise.MULTIPLY: kf.KernelType.VECTOR_MUL,
    oc.OperationElementwise.DIVIDE: kf.KernelType.VECTOR_DIV,
    
    # Activation operations
    oc.OperationActivation.RELU: kf.KernelType.RELU,
    oc.OperationActivation.SIGMOID: kf.KernelType.SIGMOID,
    oc.OperationActivation.TANH: kf.KernelType.TANH,
    oc.OperationActivation.LEAKY_RELU: kf.KernelType.LEAKY_RELU,
    oc.OperationActivation.ELU: kf.KernelType.ELU,
    oc.OperationActivation.SELU: kf.KernelType.SELU,
    oc.OperationActivation.PARAMETRIC_RELU: kf.KernelType.PARAMETRIC_RELU,
    oc.OperationActivation.LINEAR: kf.KernelType.LINEAR,
    
    # Normalization operations
    oc.OperationNormalization.BATCH_NORM: kf.KernelType.BATCH_NORM_SIMPLE_TRAINING,
    oc.OperationNormalization.LAYER_NORM: kf.KernelType.LAYER_NORM_SIMPLE,
    oc.OperationNormalization.RMS_NORM: kf.KernelType.RMS_NORM,
    oc.OperationNormalization.INSTANCE_NORM: kf.KernelType.INSTANCE_NORM,
}


def get_kernel_config_for_operation(operation: oc.OperationBase, name: str, **kwargs) -> fv.ValidationConfig:
    """
    Get the appropriate kernel configuration for a given operation.
    
    Args:
        operation: The operation type from operation_classification
        name: Name for the validation config
        **kwargs: Additional parameters needed for the specific operation (e.g., shape, axis)
        
    Returns:
        ValidationConfig object for the specified operation
        
    Example:
        # For elementwise operations
        config = get_kernel_config_for_operation(
            oc.OperationElementwise.ADD, 
            "vector_add_test", 
            shape=(1024,)
        )
        
        # For matrix multiplication
        config = get_kernel_config_for_operation(
            oc.OperationMatmul.LINEAR,
            "matmul_test",
            M=128, N=256, K=512
        )
        
        # For normalization with axis
        config = get_kernel_config_for_operation(
            oc.OperationNormalization.BATCH_NORM,
            "batch_norm_test",
            shape=(32, 64, 28, 28),
            axis=1
        )
    """
    if operation not in OPERATION_TO_KERNEL_CONFIG_MAP:
        raise ValueError(f"Unsupported operation: {operation}")
    
    config_generator = OPERATION_TO_KERNEL_CONFIG_MAP[operation]
    
    # Handle special cases that need specific parameters
    if operation in [oc.OperationNormalization.BATCH_NORM, oc.OperationNormalization.LAYER_NORM]:
        # These operations need axis parameter
        return config_generator(name, **kwargs)
    elif operation == oc.OperationMatmul.LINEAR:
        # Matrix multiplication needs M, N, K parameters
        return config_generator(name, **kwargs)
    else:
        # Most operations just need name and shape
        return config_generator(name, **kwargs)


def get_kernel_type_for_operation(operation: oc.OperationBase) -> kf.KernelType:
    """
    Get the kernel type directly for a given operation.
    
    Args:
        operation: The operation type from operation_classification
        
    Returns:
        KernelType enum value
    """
    if operation not in OPERATION_TO_KERNEL_TYPE_MAP:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return OPERATION_TO_KERNEL_TYPE_MAP[operation]


def create_elementwise_config(operation: oc.OperationElementwise, name: str, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    """Create validation config for elementwise operations."""
    return get_kernel_config_for_operation(operation, name, shape=shape)


def create_activation_config(operation: oc.OperationActivation, name: str, shape: Tuple[int, ...]) -> fv.ValidationConfig:
    """Create validation config for activation operations.""" 
    return get_kernel_config_for_operation(operation, name, shape=shape)


def create_normalization_config(operation: oc.OperationNormalization, name: str, 
                               shape: Tuple[int, ...], axis: Optional[Any] = None) -> fv.ValidationConfig:
    """Create validation config for normalization operations."""
    if operation == oc.OperationNormalization.BATCH_NORM:
        axis = axis if axis is not None else -1
        return get_kernel_config_for_operation(operation, name, shape=shape, axis=axis)
    elif operation == oc.OperationNormalization.LAYER_NORM:
        axis = axis if axis is not None else (-1,)
        return get_kernel_config_for_operation(operation, name, shape=shape, axis=axis)
    else:
        return get_kernel_config_for_operation(operation, name, shape=shape)


def create_matmul_config(operation: oc.OperationMatmul, name: str, M: int, N: int, K: int) -> fv.ValidationConfig:
    """Create validation config for matrix multiplication operations."""
    if operation == oc.OperationMatmul.LINEAR:
        return kc.generate_matrix_multiply_config(name, M, N, K)
    else:
        # For other matmul operations, use the basic matrix multiply config
        return kc.generate_matrix_multiply_config(name, M, N, K)


# Convenience function to get all supported operations
def get_supported_operations() -> Dict[oc.OperationType, list]:
    """
    Get all supported operations organized by operation type.
    
    Returns:
        Dictionary mapping operation types to lists of supported operations
    """
    return {
        oc.OperationType.ELEMENTWISE: list(oc.OperationElementwise),
        oc.OperationType.ACTIVATION: list(oc.OperationActivation), 
        oc.OperationType.NORMALIZATION: list(oc.OperationNormalization),
        oc.OperationType.MATMUL: list(oc.OperationMatmul),
    }



class ModelVerification:
    def __init__(self, profile_dir: str):
        self.prediction_manager = PredictionManager()
        self.validation_manager = fv.ValidationManager(profile_dir=profile_dir)

    def add_verification_config(self, operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]):
        self.prediction_manager.add_config(operation_type, operation, shapes, operation_params)
        
        # Generate a unique name for this configuration
        config_name = f"{operation_type.value}_{operation.value}_{len(self.prediction_manager.config_list)}"
        
        # Prepare kwargs for kernel config generation based on operation type
        if operation_type == oc.OperationType.MATMUL:
            # Matrix multiplication operations need M, N, K parameters
            config_kwargs = operation_params.copy()
        else:
            # Other operations typically need shape parameter
            config_kwargs = {'shape': shapes[0] if shapes else (1,)}
            config_kwargs.update(operation_params)
        
        self.validation_manager.add_config(get_kernel_config_for_operation(operation, config_name, **config_kwargs))

    def verify(self):
        self.prediction_manager.predict()
        self.validation_manager.profile_all_packages(repeat = 10)
        self.validation_manager.parse_all_packages()
        prediction_df = self.prediction_manager.get_dataframe()
        validation_df = self.validation_manager.get_filtered_events_dataframe_for_avg_fusion_duration(save_to_file=True)
        
        # Merge the two dataframes by combining columns (rows kept same)
        merged_df = pd.concat([prediction_df, validation_df], axis=1)
        
        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'operation_type': 'Operation_Type',
            'operation': 'Operation',
            'shapes': 'Input_Shapes',
            'latency_us': 'Predicted_Latency_us',
            'kernel_name': 'Kernel_Name',
            'dur(us)': 'Actual_Duration_us'
        })
        # Calculate error percentage: ((predicted - actual) / actual) * 100
        merged_df['Error_Percentage'] = ((merged_df['Predicted_Latency_us'] - merged_df['Actual_Duration_us']) / merged_df['Actual_Duration_us']) * 100
        
        # Calculate RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((merged_df['Predicted_Latency_us'] - merged_df['Actual_Duration_us']) ** 2))
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs(merged_df['Error_Percentage']))
        # Print the metrics
        print(f"\n=== Model Verification Metrics ===")
        print(f"RMSE (Root Mean Square Error): {rmse:.2f} microseconds")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"Number of test cases: {len(merged_df)}")
        
        # Save the merged dataframe
        merged_df.to_csv(f"{self.validation_manager.profile_dir}/merged_verification_results.csv", index=False)
        
        return merged_df
