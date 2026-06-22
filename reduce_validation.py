import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc

shape_list_1d = []

for m in range(32, 4097, 32):
    shape_list_1d.append((m,))

shape_list_2d = []
for m in range(32, 1025, 32):
    for n in range(32, 1025, 32):
        shape_list_2d.append((m, n))

config_list = []
# for shape in shape_list_1d:
#     config_list.append(kc.generate_sum_reduce_config(f"product_reduce_{shape}", shape, axis = (0,)))
for shape in shape_list_2d:
    config_list.append(kc.generate_sum_reduce_config(f"product_reduce_{shape}", shape, axis = (0)))

manager = fv.ValidationManager(profile_dir="./traces/trace_sum_reduce2d0_repeat10")

for config in config_list:
    manager.add_config(config)

manager.profile_all_packages(repeat = 10)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)

config_list = []
    