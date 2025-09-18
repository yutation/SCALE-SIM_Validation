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
# M_range = [32, 40, 48, ]
# N_range = [2, 8, 32, 128, 512]
# K_range = [2, 8, 32, 128, 512]

# for M in range(128, 513, 16):
#     MNK_list.append((M, 128, 128))

# for N in range(128, 513, 16):
#     MNK_list.append((128, N, 128))

# for K in range(128, 513, 16):
#     MNK_list.append((128, 128, K))


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

manager = fv.ValidationManager(profile_dir="./traces/trace_matmul7_repeat20")

for config in config_list:
    manager.add_config(config)

manager.profile_all_packages(repeat = 10)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)
manager.write_scale_sim_topology_csv()