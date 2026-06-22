import profiling_manager as pm
import jax_kernel_functions as jkf
import jax.numpy as jnp


pm_configuration = {
    "storage_file": "./test_results2/test_add.csv",
    "storage_metadata_file": "./test_results2/test_metadata.csv",
    "append_to_metadata_file": True,
    "hardware_config": "TPUv4",
    "operator_name": "add",
    "common_operator_dimensions": None,
    "data_precision": "FP16",
    "profiler_iterations": 5,
    "random_seed": 0,
    "repo_version": "v0.0.0",
    "comment": "test",
}
manager = pm.ProfilingManagerSimpleElementwise("test", "./test_results2", pm_configuration)

for M in [128, 256, 512, 1024]:
    for N in [128, 256, 512, 1024]:
            kernel_wrapper = jkf.KernelWarpper("add", [((M,N), jnp.float16),((M,N), jnp.float16)])
            print("test_add_M{}_N{}".format(M, N))
            manager.add_profiler("test_add_M{}_N{}".format(M, N), kernel_wrapper)


# manager.profile_and_post_process_all_profilers()
manager.post_process_all_profilers()
manager.write_results()