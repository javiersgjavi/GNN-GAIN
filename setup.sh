cd docker
docker build . -t imputate_image
docker-compose up -d
container_id=$(docker ps -aqf "name=^docker_container_imputate_image")
echo "Container ID: $container_id"
docker exec -it "$container_id" bash