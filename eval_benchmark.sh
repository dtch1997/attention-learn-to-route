#!/bin/bash

GRAPH_SIZE=$1

for ID in {0..19}
do 
    for model in random_dtspms pretrained/dtspms_20
    do
	if (($ID < 10))
	then 
	    instance_id="0${ID}"
	else 
	    instance_id=$ID
	fi
	echo "$GRAPH_SIZE, $instance_id, $model"
        python eval.py data/dtspms/R${instance_id}_${GRAPH_SIZE}_2_${GRAPH_SIZE}.pkl --model $model --decode_strategy greedy
    done
done
