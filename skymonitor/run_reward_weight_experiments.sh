#!/usr/bin/env zsh

set -e

region_ids_file=${REGION_IDS_FILE:-skymonitor/clusters4_Is.txt}
scratch_root=${SCRATCH_ROOT:-./scratch/skymonitor_reward_weights}
repeats=${1:-5}

policy_type=${POLICY_TYPE:-simple}
python_bin=${PYTHON_BIN:-python}
train_timesteps=${TRAIN_TIMESTEPS:-1000000}
num_envs=${NUM_ENVS:-8}
learning_rate=${LEARNING_RATE:-1e-4}
eval_freq=${EVAL_FREQ:-10000}
eval_repeat=${EVAL_REPEAT:-5}
eval_seed=${EVAL_SEED:-888}
save_name=${SAVE_NAME:-model}
skip_existing=${SKIP_EXISTING:-1}

run_one() {
	local task_name=$1
	local num_drones=$2
	local label=$3
	local a=$4
	local b=$5
	local c=$6
	local d=$7

	for i in $(seq 1 ${repeats}); do
		local base_seed=$((1000 + i * 10))
		local logdir="${scratch_root}/${task_name}/dr${num_drones}/${label}/rep${i}"
		local result_file="${logdir}/rep${eval_repeat}x_results.json"

		if [[ "${skip_existing}" == "1" && -f "${result_file}" ]]; then
			echo "Skipping ${logdir}; ${result_file} already exists."
			continue
		fi

		echo "Running ${task_name} ${label}, drones=${num_drones}, repeat=${i}, weights=(${a}, ${b}, ${c}, ${d})"
		${python_bin} skymonitor/rl_drone.py \
			--policy-type ${policy_type} \
			--reward-type balanced \
			--balanced-reward-weights ${a} ${b} ${c} ${d} \
			--region-ids-file ${region_ids_file} \
			--train-timesteps ${train_timesteps} \
			--num-envs ${num_envs} \
			--num-drones ${num_drones} \
			--learning-rate ${learning_rate} \
			--train-seed ${base_seed} \
			--eval-freq ${eval_freq} \
			--eval-repeat ${eval_repeat} \
			--eval-seed ${eval_seed} \
			--logdir ${logdir} \
			--save-name ${save_name}
	done
}

mfd_values=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
region_mfd_values=(0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.00)
mfd_tags=(005 010 015 020 025 030 035 040 045 050)
region_mfd_tags=(045 040 035 030 025 020 015 010 005 000)

for idx in $(seq 1 ${#mfd_values[@]}); do
	run_one \
		task1 \
		10 \
		"a025_b025_mfd${mfd_tags[$idx]}_region${region_mfd_tags[$idx]}" \
		0.25 \
		0.25 \
		${mfd_values[$idx]} \
		${region_mfd_values[$idx]}
done

congestion_values=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
congestion_level_values=(0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.00)
congestion_tags=(005 010 015 020 025 030 035 040 045 050)
congestion_level_tags=(045 040 035 030 025 020 015 010 005 000)

for idx in $(seq 1 ${#congestion_values[@]}); do
	run_one \
		task2 \
		10 \
		"a${congestion_tags[$idx]}_b${congestion_level_tags[$idx]}_mfd025_region025" \
		${congestion_values[$idx]} \
		${congestion_level_values[$idx]} \
		0.25 \
		0.25
done

w1_values=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45)
w2_values=(0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05)
w1_tags=(005 010 015 020 025 030 035 040 045)
w2_tags=(045 040 035 030 025 020 015 010 005)

for idx in $(seq 1 ${#w1_values[@]}); do
	run_one \
		task3 \
		10 \
		"w1${w1_tags[$idx]}_w2${w2_tags[$idx]}" \
		${w1_values[$idx]} \
		${w1_values[$idx]} \
		${w2_values[$idx]} \
		${w2_values[$idx]}
done

for num_drones in 20 50 100; do
	for idx in $(seq 1 ${#w1_values[@]}); do
		run_one \
			task4 \
			${num_drones} \
			"w1${w1_tags[$idx]}_w2${w2_tags[$idx]}" \
			${w1_values[$idx]} \
			${w1_values[$idx]} \
			${w2_values[$idx]} \
			${w2_values[$idx]}
	done
done

${python_bin} skymonitor/plot_reward_weight_experiments.py \
	--scratch-dir ${scratch_root} \
	--output-dir ${scratch_root}/figures
