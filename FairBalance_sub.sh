#!/bin/sh


echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                     : `date +%F-%H:%M:%S`\n"
main_dir=/cbica/home/houbo/Projects/FairBalanceMachine

for dataset in tadpole toxic
do
    echo "Processing dataset $dataset..."

    for method in 1 2 3 4 5 6
    do
        echo "Running method $method..."

        for lr_prior in 0.1 0.01 0.001 0.0001
        do
            echo "learning rate for prior model $lr_prior"

            for lr_post in 0.4 0.1 0.01
            do
                echo "learning rate for post model $lr_post"
                jid=$(qsub \
                      -terse \
                      -l h_vmem=40G \
                      -l gpu \
                      -pe threaded 4\
                      -o ${main_dir}/logs/${dataset}\$JOB_NAME-\$JOB_ID.stdout \
                      -e ${main_dir}/logs/${dataset}\$JOB_NAME-\$JOB_ID.stderr \
                      ${main_dir}/FairBalance.sh -d $dataset -m $main_dir -me $method -pr $lr_prior -po $lr_post
                      )
            done
        done
    done
done