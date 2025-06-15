# Fashion Item Classification Using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying fashion items using the Fashion MNIST dataset. The system includes a complete end-to-end solution from model training to deployment as a web application using Streamlit and Docker.

## Dataset
The project utilizes the Fashion MNIST dataset, which consists of 70,000 grayscale images (60,000 for training and 10,000 for testing) across 10 different fashion categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Each image is 28x28 pixels in grayscale format.

## Project Structure
- `Fashion_MNIST_model_training.ipynb`: Jupyter notebook containing the model training code
- `fashion-mnist-end-to-end-project-main/`: Main project directory
  - `app/`: Streamlit web application
    - `main.py`: Streamlit application code
    - `Dockerfile`: Docker configuration for containerization
    - `requirements.txt`: Python dependencies
    - `trained_model/`: Directory containing the trained model
  - `test_images/`: Sample images for testing the model
  - `model_training_notebook/`: Additional training notebooks

## Technology Stack
- **Python**: Primary programming language
- **TensorFlow/Keras**: For building and training the CNN model
- **NumPy**: For numerical operations on image data
- **Streamlit**: For creating the web interface
- **Docker**: For containerizing the application
- **PIL (Python Imaging Library)**: For image processing

## Model Architecture
The CNN architecture consists of:
1. Convolutional layers with ReLU activation
2. MaxPooling layers for downsampling
3. Dense layers for classification
4. Output layer with 10 neurons (one for each fashion category)

The model achieves approximately 90% accuracy on the test dataset.

## Web Application
The Streamlit web application allows users to:
1. Upload an image of a fashion item
2. Process the image to match the format required by the model
3. Classify the image using the pre-trained CNN model
4. Display the predicted fashion category

## Deployment
The application is containerized using Docker, making it easy to deploy across different environments. The Dockerfile sets up a Python environment with all necessary dependencies and configures the Streamlit application to run on port 80.

## How to Use
1. Clone this repository
2. Navigate to the app directory: `cd fashion-mnist-end-to-end-project-main/app`
3. Build the Docker image: `docker build -t fashion-classifier .`
4. Run the Docker container: `docker run -p 8501:8501 fashion-classifier`
5. Access the application in your browser at `http://localhost:8501`

## Training Your Own Model
To train your own model, run the `Fashion_MNIST_model_training.ipynb` notebook in a Jupyter environment with TensorFlow installed. 