from typing import Any, Dict, List, Tuple
import operation_classification as oc
import linear_models as lm
from utils import DataFrameGenerator
import math



class PredictionManager:
    def __init__(self):
        self.df_generator = DataFrameGenerator()
        self.config_list = []

    def add_config(self, operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]):
        self.config_list.append([operation_type, operation, shapes, operation_params])

    def predict(self):
        for config in self.config_list:
            operation_type, operation, shapes, operation_params = config
            latency = latency_prediction(operation_type, operation, shapes, operation_params)
            # Convert latency from milliseconds to microseconds and clean enum values
            latency_microseconds = latency  # Convert ms to μs
            self.df_generator.add_data("operation_type", [operation_type.value])
            self.df_generator.add_data("operation", [operation.value])
            self.df_generator.add_data("shapes", [shapes])
            self.df_generator.add_data("latency_us", [latency_microseconds])
        return self.df_generator

    def matmul_sim_predict(self):
        for config in self.config_list:
            operation_type, operation, shapes, operation_params = config
            latency = matmul_sim_prediction(operation_type, operation, shapes, operation_params)
            # Convert latency from milliseconds to microseconds and clean enum values
            latency_microseconds = latency  # Convert ms to μs
            self.df_generator.add_data("operation_type", [operation_type.value])
            self.df_generator.add_data("operation", [operation.value])
            self.df_generator.add_data("shapes", [shapes])
            self.df_generator.add_data("latency_us", [latency_microseconds])
        return self.df_generator

    def dump_csv(self, file_path: str):
        self.df_generator.to_dataframe().to_csv(file_path, index=False)

    def get_dataframe(self):
        return self.df_generator.to_dataframe()


    



def latency_prediction(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any] = None) -> float:
    if operation_type == oc.OperationType.ELEMENTWISE:
        return latency_prediction_elementwise(operation_type, operation, shapes, operation_params)
    elif operation_type == oc.OperationType.ACTIVATION:
        return latency_prediction_activation(operation_type, operation, shapes, operation_params)
    elif operation_type == oc.OperationType.NORMALIZATION:
        return latency_prediction_normalization(operation_type, operation, shapes, operation_params)
    elif operation_type == oc.OperationType.POOLING:
        return latency_prediction_pooling(operation_type, operation, shapes, operation_params)
    elif operation_type == oc.OperationType.MATMUL:
        return latency_prediction_matmul(operation_type, operation, shapes, operation_params)
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}")

def matmul_sim_prediction(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.MATMUL
    assert operation in oc.OperationMatmul.__members__.values()
    assert len(shapes) == 2
    lhs_shape = shapes[0]
    rhs_shape = shapes[1]
    assert len(lhs_shape) == 2
    assert len(rhs_shape) == 2
    assert lhs_shape[1] == rhs_shape[0]
    m, n, k = lhs_shape[0], rhs_shape[1], rhs_shape[0]
    return lm.matmul_scale_sim_model(m, n, k)

def latency_prediction_elementwise(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.ELEMENTWISE
    assert operation in oc.OperationElementwise.__members__.values()
    assert len(shapes) == 1

    shape = shapes[0]

    if len(shape) > 2:
        #NOTE: This is a temporary solution to handle the case where the shape is greater than 2
        dim1 = math.prod(shape[:-1])
        dim2 = shape[-1]
        shape = (dim1, dim2)

    # Handle outlier cases for 2D shape
    if len(shape) == 2:
        dim1, dim2 = shape
        if(dim1 % 128 ==0 and dim2 % 128 !=0 and dim2 > 128):
            dim2 = ((dim2//128) + 1)*128
        elif(dim1 % 128 !=0 and dim2 % 128 ==0 and dim1 > 128):
            dim1 = ((dim1//128) + 1)*128
        shape = (dim1, dim2)
    
    size = 1
    for dim in shape:
        size *= dim

    if operation == oc.OperationElementwise.ADD:
        if len(shape) == 1:
            return lm.linear_model_elementwise_add_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_elementwise_add_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationElementwise.SUBTRACT:
        if len(shape) == 1:
            return lm.linear_model_elementwise_subtract_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_elementwise_subtract_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationElementwise.MULTIPLY:
        if len(shape) == 1:
            return lm.linear_model_elementwise_multiply_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_elementwise_multiply_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationElementwise.DIVIDE:
        if len(shape) == 1:
            return lm.linear_model_elementwise_divide_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_elementwise_divide_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    else:
        # TODO: temporary use add model
        # I think the rest is the bit oeprations, which may close to add
        if len(shape) == 1:
                    return lm.linear_model_elementwise_add_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_elementwise_add_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")



def latency_prediction_activation(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.ACTIVATION
    assert operation in oc.OperationActivation.__members__.values()
    assert len(shapes) == 1

    shape = shapes[0]
    
    if len(shape) > 2:
        # NOTE: This is a temporary solution to handle the case where the shape is greater than 2
        dim1 = math.prod(shape[:-1])
        dim2 = shape[-1]
        shape = (dim1, dim2)

    # Handle outlier cases for 2D shape (similar to elementwise operations)
    if len(shape) == 2:
        dim1, dim2 = shape
        if(dim1 % 128 == 0 and dim2 % 128 != 0 and dim2 > 128):
            dim2 = ((dim2//128) + 1)*128
        elif(dim1 % 128 != 0 and dim2 % 128 == 0 and dim1 > 128):
            dim1 = ((dim1//128) + 1)*128
        shape = (dim1, dim2)
    
    size = 1
    for dim in shape:
        size *= dim

    # Use specific models based on operation and dimension
    if operation == oc.OperationActivation.BINARY:
        if len(shape) == 1:
            return lm.linear_model_activation_binary_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_binary_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.ELU:
        if len(shape) == 1:
            return lm.linear_model_activation_elu_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_elu_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.LEAKY_RELU:
        if len(shape) == 1:
            return lm.linear_model_activation_leaky_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_leaky_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.LINEAR:
        if len(shape) == 1:
            return lm.linear_model_activation_linear_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_linear_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.PARAMETRIC_RELU:
        if len(shape) == 1:
            return lm.linear_model_activation_parametric_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_parametric_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.RELU:
        if len(shape) == 1:
            return lm.linear_model_activation_relu_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_relu_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.SELU:
        if len(shape) == 1:
            return lm.linear_model_activation_selu_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_selu_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.SIGMOID:
        if len(shape) == 1:
            return lm.linear_model_activation_sigmoid_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_sigmoid_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationActivation.TANH:
        if len(shape) == 1:
            return lm.linear_model_activation_tanh_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_activation_tanh_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def latency_prediction_normalization(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.NORMALIZATION
    assert operation in oc.OperationNormalization.__members__.values()

    shape = shapes[0]
    
    assert len(shape) >=2
    # NOTE: This is a temporary solution to handle the case where the shape is greater than 2
    
    size = 1
    for dim in shape:
        size *= dim

    # Use specific models based on operation and dimension
    if operation == oc.OperationNormalization.LAYER_NORM:
            return lm.linear_model_normalization_layer_norm_2d(size)
    elif operation == oc.OperationNormalization.RMS_NORM:
            return lm.linear_model_normalization_rms_norm_2d(size)
    elif operation == oc.OperationNormalization.BATCH_NORM:
        return lm.linear_model_normalization_layer_norm_2d(size) 

    else:
        raise ValueError(f"Unsupported operation: {operation}")

def latency_prediction_pooling(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.POOLING
    assert operation in oc.OperationPooling.__members__.values()
    shape = shapes[0]
    assert len(shape) == 4

    # Calculate total size from shape
    size = 1
    for dim in shape:
        size *= dim

    if operation == oc.OperationPooling.MAX_POOLING:
        if shape[1] %128 == 0 and (shape[2] %256 == 128 or shape[3] %128 == 128):
            return lm.linear_model_pooling_max_pooling_s1(size)
        elif shape[2] > 768 and shape[2] <= 1024 and shape[3] >= 768 and shape[3] <= 896:
            return lm.linear_model_pooling_max_pooling_s2(size)
        else:
            return lm.linear_model_pooling_max_pooling(size)
    elif operation == oc.OperationPooling.AVG_POOLING:
        # Use max pooling model scaled down for avg pooling (typically faster)
        return lm.linear_model_pooling_max_pooling(size) * 0.8
    else:
        raise ValueError(f"Unsupported operation: {operation}")

def latency_prediction_matmul(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.MATMUL
    assert operation in oc.OperationMatmul.__members__.values()
    assert len(shapes) == 2
    lhs_shape = shapes[0]
    rhs_shape = shapes[1]
    assert len(lhs_shape) == 2
    assert len(rhs_shape) == 2
    assert lhs_shape[1] == rhs_shape[0]
    m, n, k = lhs_shape[0], rhs_shape[1], rhs_shape[0]
    return lm.linear_model_matmul_0(m, n, k)





