

import math

"""
Elementwise Operations
"""
def linear_model_elementwise_add_1d(size: int) -> float:
    return 0.00002 * size + 0.892433

def linear_model_elementwise_add_2d(size: int) -> float:
    return 0.00001763* size + 0.949103

def linear_model_elementwise_subtract_1d(size: int) -> float:
    return 0.00002284 * size + 0.875072

def linear_model_elementwise_subtract_2d(size: int) -> float:
    return 0.00001766 * size + 0.950563

def linear_model_elementwise_multiply_1d(size: int) -> float:
    return 0.00001945 * size + 0.894560

def linear_model_elementwise_multiply_2d(size: int) -> float:
    return 0.00001766* size + 0.936941

def linear_model_elementwise_divide_1d(size: int) -> float:
    return 0.00002195 * size + 0.906242

def linear_model_elementwise_divide_2d(size: int) -> float:
    return 0.00002066 * size + 0.938534


"""
Normalization Function Models
Based on operation_dimension_analysis_report.txt
"""

# Layer normalization models
def linear_model_normalization_layer_norm_2d(size: int) -> float:
    return 0.00000293 * size + 1.393463

# RMS normalization models  
def linear_model_normalization_rms_norm_2d(size: int) -> float:
    return 0.00000323 * size + 1.799989

# Batch normalization models
def linear_model_normalization_batch_norm_2d(size: int) -> float:
    return 0.0000005915 * size + 23.970025


"""
Activation Function Models
Based on operation_dimension_analysis_report.txt
"""
# Binary activation models
def linear_model_activation_binary_1d(size: int) -> float:
    return 0.00001220 * size + 0.870054

def linear_model_activation_binary_2d(size: int) -> float:
    return 0.00000872 * size + 1.209717

# ELU activation models  
def linear_model_activation_elu_1d(size: int) -> float:
    return 0.00001524 * size + 0.887225

def linear_model_activation_elu_2d(size: int) -> float:
    return 0.00001098 * size + 1.200012

# Leaky ReLU activation models
def linear_model_activation_leaky_1d(size: int) -> float:
    return 0.00001421 * size + 0.879131

def linear_model_activation_leaky_2d(size: int) -> float:
    return 0.00001040 * size + 1.279116

# Linear activation models
def linear_model_activation_linear_1d(size: int) -> float:
    return 0.00001407 * size + 0.877689

def linear_model_activation_linear_2d(size: int) -> float:
    return 0.00001113 * size + 1.124353

# Parametric ReLU activation models
def linear_model_activation_parametric_1d(size: int) -> float:
    return 0.00001352 * size + 0.879288

def linear_model_activation_parametric_2d(size: int) -> float:
    return 0.00001054 * size + 1.229131

# ReLU activation models
def linear_model_activation_relu_1d(size: int) -> float:
    return 0.00001291 * size + 0.881475

def linear_model_activation_relu_2d(size: int) -> float:
    return 0.00001088 * size + 1.216916

# SELU activation models
def linear_model_activation_selu_1d(size: int) -> float:
    return 0.00001389 * size + 0.893340

def linear_model_activation_selu_2d(size: int) -> float:
    return 0.00001136 * size + 1.197927

# Sigmoid activation models
def linear_model_activation_sigmoid_1d(size: int) -> float:
    return 0.00001633 * size + 0.897938

def linear_model_activation_sigmoid_2d(size: int) -> float:
    return 0.00001357 * size + 1.276542

# Tanh activation models
def linear_model_activation_tanh_1d(size: int) -> float:
    return 0.00001568 * size + 0.881633

def linear_model_activation_tanh_2d(size: int) -> float:
    return 0.00001218 * size + 0.932870


"""
Pooling Function Models
"""
# Max pooling models
def linear_model_pooling_max_pooling_s1(size: int) -> float:
    return 0.0000014714 * size + 1.50135

def linear_model_pooling_max_pooling_s2(size: int) -> float:
    return 0.000003813 * size + 0.849000

def linear_model_pooling_max_pooling(size: int) -> float:
    return 0.0000021834 * size + 2.962763
    

"""
Matrix Multiplication Models
"""
def matmul_scale_sim_model(m: int, n: int, k: int, systolic_array_size: int = 128) -> int:
    v1 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
    m, n = n, m
    v2 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
    return min(v1, v2)

def linear_model_matmul_0(m: int, n: int, k: int) -> int:
    cycles = matmul_scale_sim_model(m, n, k)

    if m < 128 and n < 128 and k < 128:
        return 0.002762 * cycles + 0.059902
    elif m < 1024 and n < 1024 and k < 1024:
        return 0.00036 * cycles + 0.780070
    else:
        return 0.00020227 * cycles + 29.721747

def linear_model_matmul(m: int, n: int, k: int) -> int:
    cycles = matmul_scale_sim_model(m, n, k)

    if m < 128 and n < 128 and k < 128:
        return 0.002762 * cycles + 0.059902
    else:
        return 2**(-6.802) * cycles**(0.725)

    


    