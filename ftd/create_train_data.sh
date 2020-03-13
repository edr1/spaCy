#!/bin/bash
#in_file="FTD_autores.txt"
#out_file="FTD_autores_train.dat.py"
#entity_type="AUTOR"

create_train_data () {
    echo "Creating training data"
    echo "in_file: $1"
    echo "out_file: $2"
    echo "entity_type: $3"
    echo "TRAIN_DATA_$3_1E = [" > $2
    while read line
    do 
        tot="$(echo $line | wc -L)"
        echo "(\"$line\",{\"entities\":[(0,"$tot",'$3')]})," >> $2
    done < ./$1
    echo "]" >> $2
}

main () {
    if [ $# -eq 3 ]; then
        create_train_data $1 $2 $3
    else
        echo "Please psupply 3 parameters... "
        echo "in_file=FTD_autores.txt"
        echo "out_file=FTD_autores_dat.py"
        echo "entity_type=AUTOR"
    fi
}
main $1 $2 $3
