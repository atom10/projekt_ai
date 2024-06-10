#!/bin/bash

# Start the services in detached mode
docker-compose up -d --build --remove-orphans

# Attach to the python_app container
docker attach python_app

docker-compose stop
