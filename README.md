# TritonVision

This project demonstrates the deployment of a machine learning model using NVIDIA Triton Inference Server on an RTX 3060 12GB GPU. It includes setting up the Triton server, preparing the model for deployment, making inference requests, and preprocessing images.

Prerequisites
Anaconda (For Virtual Environment, Install CUDA and cuDNN from conda)

Docker installed on your system

NVIDIA Docker support (check for NVIDIA driver and Docker version compatibility)

Python 3.11

tritonclient Python package

PIL (Python Imaging Library)

scipy

Installation
Install Docker: Follow the official Docker installation guide to install Docker on your system.

Install NVIDIA Triton Server: Pull the Triton Server Docker image.

Clone the Project Repository: Clone the project repository from GitHub to your local machine.

Download the MobileNetV2 Model: Download the MobileNetV2 model and move it to the appropriate directory in your Triton model repository.

Model Configuration: Prepare the config.pbtxt file, which defines the model configuration for Triton Inference Server.

Docker Compose Setup: Create a docker-compose.yml file to manage the Triton Server container with Docker Compose.

Environment Variables: Create a .env file to store environment variables for Docker Compose.

Running Docker Compose: Start the Triton Server using Docker Compose and verify the setup by checking logs to ensure everything is running correctly.

Inference Script
To make inference requests to the Triton server, you can preprocess images and send them for inference. Ensure the preprocessing includes resizing, normalizing, and reshaping the image before sending it to the server. The results from the inference will display the top 5 predictions with their corresponding scores.

Conclusion
This documentation provides a comprehensive guide to deploying a machine learning model using NVIDIA Triton Inference Server. It covers downloading and configuring the model, setting up and running the server, and making inference requests with preprocessing. By following these steps, you can effectively deploy and utilize machine learning models with Triton Server.
