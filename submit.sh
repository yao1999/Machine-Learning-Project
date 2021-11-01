#!/bin/bash

# first time use : https://noklam-data.medium.com/make-your-kaggle-submissions-with-kaggle-official-api-f49093c04f8a

COMPETITIONNAME="digit-recognizer"
# FILECHOOSE=$1
FILENAME=""

if [ $1 -eq 1 ]
then
    FILENAME="random_forest.csv"
elif [ $1 -eq 2 ]
then
    FILENAME="knn.csv"
fi

# if [[ -f "$FILENAME" ]]; then
#     echo "$FILENAME exists."
# fi

# kaggle competitions submissions  -c digit-recognizer

# kaggle competitions submit [-h] -c "$COMPETITIONNAME" -f "$FILENAME" -m MESSAGE
kaggle competitions submit [-h] -c digit-recognizer -f knn.csv -m MESSAGE