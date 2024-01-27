

# Docker build command
docker build . -t upload-service


# Run APP
docker run -p 8080:8080 -v C:/app/uploads upload-service
