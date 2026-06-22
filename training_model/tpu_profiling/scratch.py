import profiling_manager as pm
import jax_kernel_functions as jkf
import jax.numpy as jnp


pm_configuration = {
    "storage_file": "./test_results/test_matrix_multiply.csv",
    "storage_metadata_file": "./test_results/test_metadata.csv",
    "append_to_metadata_file": True,
    "hardware_config": "TPUv4",
    "operator_name": "matrix_multiply",
    "common_operator_dimensions": None,
    "data_precision": "FP16",
    "profiler_iterations": 10,
    "random_seed": 0,
    "repo_version": "v0.0.0",
    "comment": "test",
}
manager = pm.ProfilingManagerV0("test", "./test_results", pm_configuration)

for M in [128, 256, 512, 1024]:
    for N in [128, 256, 512, 1024]:
        for K in [128, 256, 512, 1024]:
            kernel_wrapper = jkf.KernelWarpper("matrix_multiply", [((M, K), jnp.float16), ((K, N), jnp.float16)])
            print("test_matmul_M{}_N{}_K{}".format(M, N, K))
            manager.add_profiler("test_matmul_M{}_N{}_K{}".format(M, N, K), kernel_wrapper)


manager.profile_and_post_process_all_profilers()
manager.write_results()