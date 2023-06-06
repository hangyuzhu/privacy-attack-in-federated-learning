cd ..
python fed_main.py --strategy fedavg --num_rounds 200 --num_epochs 2 --lr 0.1 --total_clients 20 --model cnn --iid --save_results
python fed_main.py --strategy fedavg --num_rounds 200 --num_epochs 2 --lr 0.1 --total_clients 20 --model cnn --beta 0.5 --save_results
python fed_main.py --strategy fedavg --num_rounds 200 --num_epochs 2 --lr 0.1 --total_clients 20 --model resnet18 --iid --save_results
python fed_main.py --strategy fedavg --num_rounds 200 --num_epochs 2 --lr 0.1 --total_clients 20 --model resnet18 --beta 0.5 --save_results
