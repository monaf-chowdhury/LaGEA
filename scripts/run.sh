#!/bin/bash


# Script to reproduce results
envs=(
    "button-press-topdown-v2-goal-observable"
    "push-v2-goal-observable"
    "door-open-v2-goal-observable"
    "drawer-open-v2-goal-observable"
    "pick-place-v2-goal-observable"

    "window-open-v2-goal-hidden"
    "window-close-v2-goal-hidden"
    "reach-v2-goal-hidden"
    "push-v2-goal-hidden"
    "drawer-close-v2-goal-hidden"
)


for seed in 42
do
    for env in ${envs[*]}
    do
        python main.py \
        --config.env_name=$env \
        --config.exp_name=lagea \
        --config.seed=$seed \
        --config.gap=10 \
        --config.expl_noise=0.2 \
        --config.embed_buffer_size=20000
    done
done
