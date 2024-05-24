DATAPATH="../federated_learning/data/"

cd ..
cd ..

# run robbing the fed
# non-iid
python server_attack.py \
    --attack rtf \
    --model resnet18 \
    --imprint \
    --data_path $DATAPATH \
    --dataset cifar10 \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 10 \
    --num_rounds 1 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --device cuda \
    --save_results

# run robbing the fed
# iid
python server_attack.py \
    --attack rtf \
    --model resnet18 \
    --imprint \
    --data_path $DATAPATH \
    --dataset cifar10 \
    --iid \
    --total_clients 10 \
    --num_rounds 1 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --device cuda \
    --save_results