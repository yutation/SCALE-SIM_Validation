import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc



shape_list = []
# M_range = [32, 40, 48, ]
# N_range = [2, 8, 32, 128, 512]
# K_range = [2, 8, 32, 128, 512]

# for M in range(128, 513, 16):
#     MNK_list.append((M, 128, 128))

# for N in range(128, 513, 16):
#     MNK_list.append((128, N, 128))

# for K in range(128, 513, 16):
#     MNK_list.append((128, 128, K))


for M in range(128, 8193, 128):
            shape_list.append((M,))
print(len(shape_list))

for M in range(32, 1025,32):
    for N in range(32, 1025, 32):
        if M*N > 8192:
            continue
        shape_list.append((M, N))
print(len(shape_list))

for M in range(8, 256, 8):
    for N in range(8, 256, 8):
        for K in range(8, 256, 8):
            if M*N*K > 8192:
                continue
            shape_list.append((M, N, K))
print(len(shape_list))

config_list = []
for shape in shape_list:
    config_name = f"add_{shape}"
    config_list.append(kc.generate_vector_mul_config(config_name, shape))

manager = fv.ValidationManager(profile_dir="./trace_mul_repeat20")

for config in config_list:
    manager.add_config(config)

manager.profile_all_packages(repeat = 20)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)