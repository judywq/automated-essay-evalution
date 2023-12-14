# __all__ = [
#     "config_set_fa_float",
#     "config_set_f1_float",
#     "config_set_f2_float",
#     "config_set_fa_int",
#     "config_set_f1_int",
#     "config_set_f2_int",
#     "config_unit_gpt_3_5_turbo",
#     "config_unit_gpt_4_turbo",
# ]

config_unit_base = "./configs/config.base.json"
config_unit_form_1 = "./configs/config.form.1.json"
config_unit_form_2 = "./configs/config.form.2.json"
config_unit_form_all = "./configs/config.form.all.json"
config_unit_int = "./configs/config.type.int.json"
config_unit_float = "./configs/config.type.float.json"
config_unit_gpt_3_5_turbo = "./configs/config.gpt3.5-turbo.json"
config_unit_gpt_4_turbo = "./configs/config.gpt4-turbo.json"

config_unit_20231202 = "./configs/config.20231202.json"
config_unit_20231203 = "./configs/config.20231203.json"
config_unit_20231208 = "./configs/config.20231208.json"

config_set_fa_float = [config_unit_base, config_unit_form_all, config_unit_float]
config_set_f1_float = [config_unit_base, config_unit_form_1, config_unit_float]
config_set_f2_float = [config_unit_base, config_unit_form_2, config_unit_float]
config_set_fa_int = [config_unit_base, config_unit_form_all, config_unit_int]
config_set_f1_int = [config_unit_base, config_unit_form_1, config_unit_int]
config_set_f2_int = [config_unit_base, config_unit_form_2, config_unit_int]

# config_list = [
#     config_set_f1_float,
#     # config_set_f1_int,
#     config_set_f2_float,
#     # config_set_f2_int,
#     config_set_fa_float,
#     # config_set_fa_int,
# ]

config_list = [
    config_set_f1_float,
    config_set_f1_int,
    config_set_f2_float,
    config_set_f2_int,
    config_set_fa_float,
    config_set_fa_int,
]

variation = config_unit_20231202
# variation = config_unit_20231203
# variation = config_unit_20231208

for confitset in config_list:
    confitset.append(variation)
    