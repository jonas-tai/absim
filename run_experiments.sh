# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 10000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5




PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_10" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_10" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_10" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_10" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_10" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_10" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_10" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5




PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_20" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_20" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_20" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_20" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_20" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_20" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_20" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5




PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_30" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_30" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_30" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_30" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "ARS_30" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_30" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "ARS_30" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5



PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "random" --test_service_time_model "random.expovariate" --batch_size 16 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "random" --test_service_time_model "pareto" --batch_size 16 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "random" --test_service_time_model "random.expovariate" --batch_size 32 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "random" --test_service_time_model "pareto" --batch_size 32 & echo $! >> pids.txt
sleep 5

PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 15000 --train_policy "random" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "random" --test_service_time_model "random.expovariate" --batch_size 64 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --from_expert_data_epoch_size 30000 --train_policy "random" --test_service_time_model "pareto" --batch_size 64 & echo $! >> pids.txt
sleep 5

