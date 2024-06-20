# Bangkit Capstone Project - Edusign (Django ML API)

This is the translation service backend for the Edusign application. This service acts as the integrator between the machine learning model and main backend service.

# Not accepting outside party contribution

This project is created to fulfill the requirements of Bangkit Academy 2024 batch 1's Capstone Project. We greatly appreciate the enthusiasm of those who would like to contribute. However, according to the rules, we are not allowed to accept help from outside parties. The public visibility of this repository is simply to allow the Bangkit team reviewer to review our project. Any and all pull request to this repository coming from outside parties will be rejected and closed. We thank you for your understanding.

# Instruction to run locally
- clone the repository
- create `.env` file with the following content
> SECRET_KEY  
> ENVIRONMENT ('production' if deployed)  
> MODEL_URL  
> ENCODER_URL  
> HAND_URL  
> POSE_URL  
> FACE_URL  
- run `python -m venv env` to create a virtual environment
- run `env\Scripts\activate.bat` to activate the virtual environment
- run `pip install -r requirements.txt` to install requirements in the virtual environment
- run `python manage.py runserver` to run the application

# API Documentation
The following is the documentation for our API.

## /translation
**Endpoint for processing video into landmark and then predicting the result with ML model**  
Method: POST  

Body:  
> link: string, required  

Response status 200 (Success):
> message: "Success message"  
> result: string 

# Deployment link
[ml-api-edusign deployment](https://edusign-ml-2frcv7abha-et.a.run.app/)
(Maybe down in the future)

# Authors
This project is developed by the Cloud Computing and Machine Learning division of C241-PS015 Bangkit Capstone Team

### Cloud Computing
1. C010D4KY1114 - Kade Satrya Noto Sadharma
2. C253D4KY1157 - Wahyu Fardiansyach

### Machine Learning
1. M006D4KY2955 – Yehezkiel Stephanus Austin
2. M006D4KY2954 – Juan Christopher Young
3. M006D4KY2953 – Haikal Irfano