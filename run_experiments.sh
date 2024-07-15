# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 10000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt

# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_10" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_10" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_20" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_20" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_30" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_30" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_40" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_40" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_50" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "ARS_50" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "random" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --offline_train_epoch_len 15000 --train_policy "random" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 4000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5




PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments/58/offline_train/data/" --offline_train_epoch_len 16000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5







PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/34/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5






PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "random.expovariate" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5


PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 4000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 8000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 32 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --offline_model "/data1/outputs/story_experiments_2/20/offline_train/data/" --offline_train_epoch_len 8000 --test_service_time_model "pareto" --batch_size 64 --offline_train_batch_size 16000 & echo $! >> pids.txt
sleep 5





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




