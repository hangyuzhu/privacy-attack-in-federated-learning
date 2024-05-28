BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run dlg
python grad_attack.py \
    --attack dlg \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --device cuda \
    --save_results

# run idlg
python grad_attack.py \
    --attack idlg \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --device cuda \
    --save_results