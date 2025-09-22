

import math


def linear_model_elementwise_add_1d(size: int) -> float:
    return 0.00002 * size + 0.892433

def linear_model_elementwise_add_2d(size: int) -> float:
    return 0.000018* size + 0.979606

def linear_model_normalization_layer_norm(size: int) -> float:
    return 0.000013* size + 13.012764


# TODO: temporary model
def linear_model_activation(size: int) -> float:
    return linear_model_elementwise_add_2d(size)



def matmul_scale_sim_model(m: int, n: int, k: int, systolic_array_size: int = 128) -> int:
    if n > m:
        m, n = n, m

    return (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)

def linear_model_matmul(m: int, n: int, k: int) -> int:
    cycles = matmul_scale_sim_model(m, n, k)

    if m < 128 and n < 128 and k < 128:
        return 0.002762 * cycles + 0.059902
    elif m < 1024 and n < 1024 and k < 1024:
        return 0.00036 * cycles + 0.780070
    else:
        return 0.0002 * cycles + 29.722

    


    