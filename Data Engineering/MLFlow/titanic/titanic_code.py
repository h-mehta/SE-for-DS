
# OS related librarires
import os
import warnings
import sys

# basic data processing libraries
import pandas as pd
import numpy as np

# XGBoost libraries
import xgboost as xg

# Sklearn libraries for metrics & train_test split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# MLFLow libraries
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

# Logger library
import logging

# Logger related :-----

# Configure the root logger

# logging.WARN : only log messages with a level of WARNING, ERROR, and CRITICAL will be displayed or saved,  
# logging.basicConfig(level=logging.WARN)

# to save logger.info messages need to set it to logging.INFO
logging.basicConfig(level=logging.INFO)

# Create a file handler and set the file path
file_handler = logging.FileHandler('titanic_run.log')

# Set the desired log format (optional)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the root logger
logging.getLogger('').addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.warning(' Start of new run ... ')

# Logger related:-----

def eval_metrics(actual, pred):
	accuracy = accuracy_score( actual, pred)
	precision = precision_score( actual, pred)
	recall = recall_score( actual, pred)
	f1 = f1_score( actual, pred)
	return accuracy, precision, recall, f1



if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	np.random.seed(42) 

	# reading in the data
	data = pd.read_csv("titanic_processed_data.csv")

	# saving the arguments read from command line into variable

	# param 1 - no of estimators
	est = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
	
	# param 2 - learning rate
	lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

	# param 3 - max_depth
	depth = int(sys.argv[3]) if len(sys.argv) > 3 else 6

	independent_variables = list(data.columns)
	independent_variables.remove("Survived")

	X_train, X_test, y_train, y_test = train_test_split( data[independent_variables], data["Survived"], test_size=0.2, random_state=42)

	xgb_r = xg.XGBRegressor(objective ='reg:squarederror', n_estimators = est, learning_rate= lr, max_depth=depth)

	# Fitting the model
	xgb_r.fit(X_train, y_train)

	pred = xgb_r.predict(X_test)

	# converting pred into levels/factors so that the classifcation metrics can work on  it
	pred = np.array( [1 if x>=0.5 else 0 for x in pred ] )
	
	accuracy, precision, recall, f1 = eval_metrics(y_test, pred)

	# Logging the parameters for MLFlow
	mlflow.log_param("n_estimators", est)
	mlflow.log_param("learning_rate", lr)
	mlflow.log_param("max_depth", depth)

	# Logging the metrics for MLFlow
	mlflow.log_metric("Accuracy", accuracy)
	mlflow.log_metric("Precision", precision)
	mlflow.log_metric("Recall", recall)
	mlflow.log_metric("F1", f1)

	# Logging for my own logs - saved as titanic_run.log
	logger.info("---------------------------------------")
	logger.info(" PARAMETERS : --- ")
	logger.info("n_estimators : %s" % est)
	logger.info("learning_rate : %s" % lr)
	logger.info("max_depth : %s " % depth)
	logger.info(" METRICS : --- ")
	logger.info("Accuracy : %s" % accuracy)
	logger.info("Precision : %s"  % precision)
	logger.info("Recall : %s" %  recall)
	logger.info("F1 : %s" % f1)
	logger.info("---------------------------------------") 





