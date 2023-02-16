#!/bin/sh


echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                     : `date +%F-%H:%M:%S`\n"
main_dir=/cbica/home/houbo/Projects/FairBalanceMachine

for dataset in tadpole toxic credit  # tadpole toxic
do
    echo "Processing dataset $dataset..."

    for method in 1 2 3 8  # 4 5 6
    do
        echo "Running method $method..."

        for model_name in FcNet4 FcNet6
        do
            echo "Model name $model_name"
            for lr_prior in 0.1 0.01 0.001 #0.1 0.01 0.001
            do
                echo "learning rate for prior model $lr_prior"

                for lr_post in 0.4 0.1 0.01
                do
                    echo "learning rate for post model $lr_post"
                    jid=$(qsub \
                          -terse \
                          -l h_vmem=40G \
                          -l gpu \
                          -o ${main_dir}/logs/${dataset}\$JOB_NAME-\$JOB_ID.stdout \
                          -e ${main_dir}/logs/${dataset}\$JOB_NAME-\$JOB_ID.stderr \
                          ${main_dir}/FairBalance.sh -d $dataset -m $main_dir \
                          -me $method -mn $model_name -pr $lr_prior -po $lr_post
                          )
                done
            done
        done
    done
done

for dataset in tadpole toxic credit  # tadpole toxic
do
    echo "Processing dataset $dataset..."

    for method in 7 9  # pure ERM and balanced ERM
    do
        echo "Running method $method..."
        for model_name in FcNet4 FcNet6
        do
            echo "Model name $model_name"
            for lr in 0.1 0.01 0.001
            do
                echo "learning rate for ERM model $lr"
                jid=$(qsub \
                      -terse \
                      -l h_vmem=40G \
                      -l gpu \
                      -o ${main_dir}/logs/${dataset}\$JOB_NAME-\$JOB_ID.stdout \
                      -e ${main_dir}/logs/${dataset}\$JOB_NAME-\$JOB_ID.stderr \
                      ${main_dir}/FairBalance_ERM.sh -d $dataset -m $main_dir -me $method -mn $model_name -lr $lr
                      )
            done
        done
    done
done