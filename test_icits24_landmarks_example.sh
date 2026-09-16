#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t icits24_landmarks_image .
sudo docker run --name icits24_landmarks_container --rm --gpus all -it -d icits24_landmarks_image bash
sudo docker exec -w /home/username/icits24_landmarks icits24_landmarks_container python test/icits24_landmarks_test.py --input-data test/example.tif --database wflw --gpu 0 --backbone EdgeNeXt --save-image
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp icits24_landmarks_container:/home/username/conda/envs/icits24/lib/python3.10/site-packages/pcr_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f icits24_landmarks_container
sudo docker image rm icits24_landmarks_image
sudo docker builder prune -a -f