train on charades-sta cpl-moe
python train.py --config-path config/charades/main_moe.json --tag cpl_moe --log_dir logs/charades_moe

train on activitynet cpl-moe
python train.py --config-path config/activitynet/main_moe.json --tag cpl_moe --log_dir logs/activitynet_moe