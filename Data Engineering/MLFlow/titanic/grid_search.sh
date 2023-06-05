

#!/bin/bash

n_estimators=(100 150 200 250 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400)

learning_rate=(0.1 0.05 0.01)

depth=(5 6 7)

for est in "${n_estimators[@]}"; do
	for lr in "${learning_rate[@]}"; do
		for d in "${depth[@]}"; do
			python titanic_code.py $est $lr $d
		done
	done
done


