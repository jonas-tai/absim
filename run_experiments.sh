# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 10000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt

# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_40" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_40" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_50" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_50" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5


# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
# sleep 5



PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/"  --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/0/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5




PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/--offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/"data" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/2/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5




PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/4/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5




PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/--offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/"" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/6/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5



PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --offline_expert_data "/dev/shm/outputs/expert_data/12/collected_training_data/" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5

