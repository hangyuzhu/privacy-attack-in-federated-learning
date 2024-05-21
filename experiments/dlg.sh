cd ..
python fed_attack.py --attack dlg --num_rounds 5 --dataset cifar10 --num_epochs 2 --rec_batch_size 1 --model cnn
python fed_attack.py --attack idlg --num_rounds 5 --dataset cifar10 --num_epochs 2 --rec_batch_size 1 --model cnn
