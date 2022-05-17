#!/bin/bash

FILE="./security_commits.txt"

while IFS= read -r commitsha
do
        echo "Jumping to commit $commitsha"
        git checkout $commitsha
        und create -languages c++ ../understand/webkit/$commitsha.udb
        und add ./ ../understand/webkit/$commitsha.udb
        und analyze ../understand/webkit/$commitsha.udb
        und settings -metrics all ../understand/webkit/$commitsha.udb
        und metrics ../understand/webkit/$commitsha.udb
        #und report ../understand/hermes/$commitsha.udb
done < "$FILE"