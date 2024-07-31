
# Reset Model

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --offline_train_batch_size 8000 --replay_memory_size 10000 --lr 0.000001 --reset_models_before_retrain & echo $! >> pids.txt
# sleep 5



# Collect expert data

# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "random.expovariate" --train_policy "ARS" & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "pareto" --train_policy "ARS" & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "random.expovariate" --train_policy "ARS_EXPLR_30" & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "pareto" --train_policy "ARS_EXPLR_30" & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "random.expovariate" --train_policy "ARS_EXPLR_50" & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "pareto" --train_policy "ARS_EXPLR_50" & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "random.expovariate" --train_policy "random" & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --test_service_time_model "pareto" --train_policy "random" & echo $! >> pids.txt
# sleep 5
#////////////////////////////////////////////////////////////////////////



## reset model

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/2/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 100 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/2/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 100 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/2/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 500 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/2/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 500 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/2/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 1000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/2/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 1000 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data_0/6/collected_training_data/" --offline_train_epoch_len 3000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 10000 & echo $! >> pids.txt
# sleep 5


## Manual Test
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --offline_train_batch_size 8000 --offline_train_epoch_len 15000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000  --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats --replay_memory_size 10000 --reset_models_before_retrain & 




# Train new base model 
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5



PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5


####
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5



PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/0/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5



PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/4/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
####






PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5




##
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 4000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 24 --offline_train_epoch_len 4000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 8000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 15000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 12 --offline_train_epoch_len 8000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/recalculate_norm/16/offline_train/data/" --epochs 6 --offline_train_epoch_len 16000 --offline_train_batch_size 16000 --test_service_time_model "pareto" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500  --expert_replay_mem_size 80000 --replay_memory_size 30000 --use_sliding_retrain_memory --recalculate_reward_stats  & echo $ ! >> pids.txt
sleep 5


# Reset Model

# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 2 --offline_train_epoch_len 35000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 1 --offline_train_epoch_len 70000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 2 --offline_train_epoch_len 35000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 1 --offline_train_epoch_len 70000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --norm_per_req_type --recalculate_reward_stats  --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5



# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 2 --offline_train_epoch_len 35000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --recalculate_reward_stats  --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 1 --offline_train_epoch_len 70000 --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --recalculate_reward_stats   --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 2 --offline_train_epoch_len 35000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --recalculate_reward_stats   --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --epochs 1 --offline_train_epoch_len 70000 --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --replay_memory_size 40000 --use_sliding_retrain_memory --norm_per_req_type  --reset_models_before_retrain & echo $ ! >> pids.txt
# sleep 5




# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/" --offline_train_epoch_len 3000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 10000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/"  --offline_train_epoch_len 5000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 20000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/"  --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 20000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/"  --offline_train_epoch_len 7000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 40000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/6/collected_training_data/"  --offline_train_epoch_len 10000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5



# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 5000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --offline_train_epoch_len 3000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 10000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --offline_train_epoch_len 5000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 20000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 20000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --offline_train_epoch_len 7000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 40000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --offline_train_epoch_len 10000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --target_update_frequency 500 --train_from_expert_data --expert_replay_mem_size 80000 --use_sliding_retrain_memory & echo $ ! >> pids.txt
# sleep 5




#////////////////////////////////////////////////////////////////////////


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/6/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 100 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/6/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 100 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/6/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 500 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/6/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 500 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/6/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 1 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 1000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/tau_new_base_model/6/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --clipping_value 10 --lr 0.000001 --replay_memory_size 15000 --offline_train_batch_size 4000 --target_update_frequency 1000 & echo $! >> pids.txt
# sleep 5





# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 5000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 5000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 5000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 5000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 10000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 10000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 10000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 10000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5



# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 --replay_memory_size 15000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.2 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.3 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.4 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.5 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.6 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.7 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.8 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.9 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 1.0 & echo $! >> pids.txt
# sleep 5



# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.2 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.3 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.4 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.5 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.6 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.7 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.8 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 0.9 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 2000 --replay_mem_retrain_expert_fraction 1.0 & echo $! >> pids.txt
# sleep 5



# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.2 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.3 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.4 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.5 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.6 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.7 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.8 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.9 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 1.0 & echo $! >> pids.txt
# sleep 5





# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.2 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.3 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.4 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.5 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.6 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.7 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.8 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.9 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 2000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 1.0 & echo $! >> pids.txt
# sleep 5




# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.0 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.2 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.3 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.4 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.5 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.6 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.7 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.8 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 0.9 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 1000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 --replay_mem_retrain_expert_fraction 1.0 & echo $! >> pids.txt
# sleep 5



# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5




# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5







# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5






# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
# sleep 5





# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/"  --test_service_time_model "random.expovariate" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "pareto" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "pareto" --batch_size 128 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "pareto" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/8/collected_training_data/" --test_service_time_model "pareto" --batch_size 256 & echo $! >> pids.txt
# sleep 5




# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "pareto" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "pareto" --batch_size 128 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "pareto" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/10/collected_training_data/" --test_service_time_model "pareto" --batch_size 256 & echo $! >> pids.txt
# sleep 5





# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 128 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 128 & echo $! >> pids.txt
# sleep 5

# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 45000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 256 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 30000 --offline_expert_data "/data1/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 256 & echo $! >> pids.txt
# sleep 5




