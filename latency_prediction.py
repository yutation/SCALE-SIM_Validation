from typing import Any, Dict, List, Tuple
import operation_classification as oc
import linear_models as lm
from utils import DataFrameGenerator



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


def latency_prediction_elementwise(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.ELEMENTWISE
    assert operation in oc.OperationElementwise.__members__.values()
    assert len(shapes) == 1

    shape = shapes[0]
    size = 1
    for dim in shape:
        size *= dim

    operation = oc.OperationElementwise.ADD # Temporary model bypass
    if operation == oc.OperationElementwise.ADD:
        if len(shape) == 1:
            return lm.linear_model_elementwise_add_1d(size)
        elif len(shape) == 2:
            return lm.linear_model_elementwise_add_2d(size)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    elif operation == oc.OperationElementwise.SUBTRACT:
        return 1.0
    elif operation == oc.OperationElementwise.MULTIPLY:
        return 1.0
    elif operation == oc.OperationElementwise.DIVIDE:
        return 1.0
    else:
        raise ValueError(f"Unsupported operation: {operation}")



def latency_prediction_activation(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.ACTIVATION
    assert operation in oc.OperationActivation.__members__.values()
    assert len(shapes) == 1

    shape = shapes[0]
    size = 1
    for dim in shape:
        size *= dim

    return lm.linear_model_activation(size)


def latency_prediction_normalization(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    assert operation_type == oc.OperationType.NORMALIZATION
    assert operation in oc.OperationNormalization.__members__.values()
    assert len(shapes) == 1

    shape = shapes[0]
    size = 1
    for dim in shape:
        size *= dim

    # TODO: use more specific models for each normalization operation
    if operation == oc.OperationNormalization.LAYER_NORM:
        return lm.linear_model_normalization_layer_norm(size)
    elif operation == oc.OperationNormalization.BATCH_NORM:
        # Use similar model to layer norm for now (could be refined with specific data)
        return lm.linear_model_normalization_layer_norm(size) * 0.8  # Batch norm is typically faster
    elif operation == oc.OperationNormalization.RMS_NORM:
        # RMS norm is similar to layer norm but slightly more efficient
        return lm.linear_model_normalization_layer_norm(size) * 0.9
    elif operation == oc.OperationNormalization.INSTANCE_NORM:
        # Instance norm operates per sample, similar overhead
        return lm.linear_model_normalization_layer_norm(size) * 1.1
    elif operation == oc.OperationNormalization.GROUP_NORM:
        # Group norm has additional grouping overhead
        return lm.linear_model_normalization_layer_norm(size) * 1.2
    else:
        raise ValueError(f"Unsupported operation: {operation}")

def latency_prediction_pooling(operation_type: oc.OperationType, operation: oc.OperationBase,  shapes: List[Tuple[int, ...]],  operation_params: Dict[str, Any]) -> float:
    pass

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
    return lm.linear_model_matmul(m, n, k)





