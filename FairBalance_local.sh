#!/bin/sh

echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                     : `date +%F-%H:%M:%S`\n"
main_dir=/Users/bojian/Documents/Research/FAMS-main
cd $main_dir
source activate FMAS-main


#for dataset in credit  # tadpole toxic
#do
#    echo "Processing dataset $dataset..."
#
#    for method in 1 2 3 8  # 4 5 6
#    do
#        echo "Running method $method..."
#
#        for model_name in FcNet4  # FcNet6
#        do
#            echo "Model name $model_name"
#            for lr_prior in 0.1 0.01 0.001 #0.1 0.01 0.001
#            do
#                echo "learning rate for prior model $lr_prior"
#
#                for lr_post in 0.4 0.1 0.01
#                do
#                    echo "learning rate for post model $lr_post"
#                    nohup python -u all_train.py --dataset $dataset --model_name $model_name \
#                                                 --method $method --lr_prior $lr_prior --lr_post $lr_post \
#                    >${main_dir}/logs/${dataset}/LOCAL/method_${method}_${model_name}_lr_prior_${lr_prior}_lr_post_${lr_post}.log 2>&1 &
#                    echo "-------------------------------------"
#                done
#            done
#        done
#    done
#done

for dataset in credit  # tadpole toxic
do
    echo "Processing dataset $dataset..."

    for method in 7 9  # pure ERM and balanced ERM
    do
        echo "Running method $method..."
        for model_name in FcNet4  # FcNet6
        do
            echo "Model name $model_name"
            for lr in 0.1 0.01 0.001
            do
                echo "learning rate for ERM model $lr"
                nohup python -u all_train.py --dataset $dataset --model_name $model_name \
                                             --method $method --lr $lr \
                >${main_dir}/logs/${dataset}/LOCAL/method_${method}_${model_name}_lr_${lr}.log 2>&1 &
            done
        done
    done
done
