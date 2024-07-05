# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 10000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
# sleep 5
# PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt

PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --num_perm 5 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --num_perm 5 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "pareto" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "pareto" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "pareto" --num_perm 5 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "pareto" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "pareto" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 5000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "pareto" --num_perm 5 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "random.expovariate" --num_perm 5 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "random.expovariate" --num_perm 5 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "pareto" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "pareto" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 8000 --test_service_time_model "pareto" --num_perm 5 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "pareto" --num_perm 1 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "pareto" --num_perm 3 & echo $! >> pids.txt
sleep 5
PYTHONPATH=./ python3 simulations/experiment.py --replay_memory_size 20000 --offline_model "/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data" --offline_train_batch_size 16000 --test_service_time_model "pareto" --num_perm 5 & echo $! >> pids.txt
