#!/bin/bash
# echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t icits24_landmarks_image .
sudo docker volume create --name icits24_landmarks_volume
sudo docker run --name icits24_landmarks_container -v icits24_landmarks_volume:/home/username --rm -it -d icits24_landmarks_image bash
sudo docker exec -w /home/username/ icits24_landmarks_container python images_framework/alignment/icits24_landmarks/test/icits24_landmarks_test.py --input-data images_framework/alignment/icits24_landmarks/test/example.tif --database wflw --gpu 0 --save-image
sudo docker stop icits24_landmarks_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/icits24_landmarks_volume/_data/output/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
