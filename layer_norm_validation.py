import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc
import kernel_functions as kf



shape_list = []



# for N in [1, 2, 4, 8, 16, 32]:
#     for L in [128,256,512,1024,2048,4096]:
#         for H in [512,768,1024,2048,4096,8192]:
#             shape_list.append((N, L, H))

for N in range(2, 14, 2):
    for L in range(128, 1025, 128):
        for H in range(512, 4097, 512):
            shape_list.append((N, L, H))

print(len(shape_list))



config_list = []

for shape in shape_list:
    shape_str = str(shape).replace(" ", "")
    config_name = f"layer_norm_{shape_str}"
    config_list.append(kc.generate_layer_norm_config(config_name, shape, axis = (2,)))

manager = fv.ValidationManager(profile_dir="./traces/trace_layer_norm2_repeat10")

for config in config_list:
    manager.add_config(config)

manager.profile_all_packages(repeat = 10)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)