#!/bin/sh


parse()
{
	while [ -n "$1" ];
	do
		case $1 in
            -d)
                dataset=$2;
                shift 2;;
            -m)
              main_dir=$2;
              shift 2;;
            -me)
              method=$2;
              shift 2;;
            -pr)
              lr_prior=$2;
              shift 2;;
            -po)
              lr_post=$2;
              shift 2;;
		esac
	done
}

if [ $# -lt 1 ]
then
	help
fi

## Reading arguments
parse $*

source activate FairBalance

cd $main_dir
python -u all_train.py --config EXPS/${dataset}_template.yml \
--method $method --lr_prior $lr_prior --lr_post $lr_post \
>${main_dir}/logs/${dataset}/CUBIC/method_${method}_lr_prior_${lr_prior}_lr_post_${lr_post}.log 2>&1