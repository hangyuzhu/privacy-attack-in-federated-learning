BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run ggl
# non-iid
python server_attack.py \
    --attack ggl \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 10 \
    --num_rounds 30 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 25000 \
    --device cuda \
    --save_results