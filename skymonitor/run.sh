#!/usr/bin/env zsh

region_ids_file=skymonitor/clusters4_Is.txt;
reward_types=(congestion_level balanced);

# those trivial agents doesn't learn, so we just pass mfd as a placeholder for reward type

# python skymonitor/monitor_env.py --agent-type random --reward-type mfd --region-ids-file ${region_ids_file} --logdir ./scratch/drone_monitor_random_mfd;

# python skymonitor/monitor_env.py --agent-type static --reward-type mfd --region-ids-file ${region_ids_file} --logdir ./scratch/drone_monitor_static_mfd;

# python skymonitor/monitor_env.py --agent-type sweeping --reward-type mfd --region-ids-file ${region_ids_file} --logdir ./scratch/drone_monitor_sweeping_mfd;

repeats=${1:-5};

echo "Running PPO policies for ${repeats} repeats..."

for reward_type in "${reward_types[@]}"; do
    for i in $(seq 1 ${repeats}); do
        base_seed=$((1000 + i * 10))

        python skymonitor/rl_drone.py --policy-type simple --reward-type ${reward_type} --region-ids-file ${region_ids_file} --train-seed ${base_seed} --logdir ./scratch/drone_monitor_ppo_simple_${reward_type}_${i};

    done
done

python skymonitor/summarize_monitor_results.py --output scratch/monitor_results_summary.md
