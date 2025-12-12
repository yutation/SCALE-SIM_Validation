import jax
import jax.numpy as jnp
import flexible_validation as fv

def generate_matrix_multiply_config(name: str, M, N, K) -> fv.ValidationConfig:
    return fv.ValidationConfig(
        name=name,
        kernel_type=fv.KernelType.MATRIX_MULTIPLY,
        inputs=[((M, K), jnp.float16),
                ((K, N), jnp.float16)]
    )



MNK_list = []
for M in range(32, 129, 16):
    for N in range(32, 129, 16):
        for K in range(32, 129, 16):
            MNK_list.append((M, N, K))
print(len(MNK_list))

config_list = []
for MNK in MNK_list:
    config_name = f"matmul_{MNK[0]}x{MNK[1]}x{MNK[2]}"
    config_list.append(generate_matrix_multiply_config(config_name, MNK[0], MNK[1], MNK[2]))

manager = fv.ValidationManager(profile_dir="./traces/matmul_128")

for config in config_list:
    manager.add_config(config)

# manager.profile_all_packages(repeat = 20)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)
manager.write_scale_sim_topology_csv()


# 1024
MNK_list = []
for M in range(128, 1025, 128):
    for N in range(128, 1025, 128):
        for K in range(128, 1025, 128):
            MNK_list.append((M, N, K))
print(len(MNK_list))

config_list = []
for MNK in MNK_list:
    config_name = f"matmul_{MNK[0]}x{MNK[1]}x{MNK[2]}"
    config_list.append(generate_matrix_multiply_config(config_name, MNK[0], MNK[1], MNK[2]))

manager = fv.ValidationManager(profile_dir="./traces/matmul_1024")

for config in config_list:
    manager.add_config(config)

# manager.profile_all_packages(repeat = 15)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)
manager.write_scale_sim_topology_csv()




#  4096
MNK_list = []
for M in range(1024, 4097, 512):
    for N in range(1024, 4097, 512):
        for K in range(1024, 4097, 512):
            MNK_list.append((M, N, K))
print(len(MNK_list))
# print(MNK_list)



config_list = []
for MNK in MNK_list:
    config_name = f"matmul_{MNK[0]}x{MNK[1]}x{MNK[2]}"
    config_list.append(generate_matrix_multiply_config(config_name, MNK[0], MNK[1], MNK[2]))

manager = fv.ValidationManager(profile_dir="./traces/matmul_4096")

for config in config_list:
    manager.add_config(config)

# manager.profile_all_packages(repeat = 10)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)
manager.write_scale_sim_topology_csv()