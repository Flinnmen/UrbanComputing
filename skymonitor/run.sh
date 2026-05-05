#!/usr/bin/env zsh

reward_type=congestion;

python skymonitor/monitor_env.py --agent-type random --reward-type ${reward_type} --logdir ./scratch/drone_monitor_random_${reward_type};

python skymonitor/monitor_env.py --agent-type static --reward-type ${reward_type} --logdir ./scratch/drone_monitor_static_${reward_type};

python skymonitor/monitor_env.py --agent-type sweeping --reward-type ${reward_type} --logdir ./scratch/drone_monitor_sweeping_${reward_type};

repeats=${1:-5};

echo "Running PPO policies for ${repeats} repeats..."

for i in $(seq 1 ${repeats}); do
    base_seed=$((1000 + i * 10))

    python skymonitor/rl_drone.py --policy-type simple --reward-type ${reward_type} --train-seed ${base_seed} --logdir ./scratch/drone_monitor_ppo_simple_${reward_type}_${i};

done
