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
            -mn)
              model_name=$2;
              shift 2;;
            -lr)
              lr=$2;
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
--method $method --model_name $model_name --lr $lr \
>${main_dir}/logs/${dataset}/CUBIC/method_${method}_${model_name}_lr_${lr}.log 2>&1