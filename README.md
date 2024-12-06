# Predicting Wine Quality with Machine Learning in Python
## Project Overview
This project demonstrates how to use machine learning to predict the quality of wine based on its chemical properties. 
The dataset for this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). 
A linear regression model is used to predict the wine quality score, allowing for insights into how specific features influence the quality.

## Directory Structure
```commandline
project-root/
│
├── dataset/
│   └── WineQuality-White.csv       # Dataset file with white wine data
│
├── src/
│   └── Wine-Quality.py             # Python script for training the model and making predictions
│
├── README.md                       # Project documentation
├── requirements.txt                # List of Python dependencies
```

## How to Use the Project

### Clone the Repository
To start using the project, clone the repository to your local system:
```commandline
git clone https://github.com/MeghanaThatikonda/Prediction-of-Wine-Quality-using-Machine-Learning.git
cd Prediction-of-Wine-Quality-using-Machine-Learning
```
### Prerequisites
Ensure you have Python 3.7 or higher installed.
Install the required Python packages by running:
```bash
pip3 install -r requirements.txt  
```
### Running the Script
```bash  
python3 src/Wine-Quality.py  
```  

During the execution, the script will:
* Train a linear regression model on the wine dataset.
* Evaluate the model on both training and test sets, displaying performance metrics like RMSE.
* Predict the quality of predefined wine samples based on chemical properties.
* Allow you to input values for the wine sample's features and predict its quality interactively.
* Follow the prompts to enter the chemical properties of a wine sample. The script will display the predicted quality score based on the model.
