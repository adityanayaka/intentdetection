1



#create a repo on git 
#create hello_word.py file 

##in terminal of VS CODE
git init
git add hello_world.py
git commit -m "Initial commit with hello world python script"
echo "venv/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "data/*.csv" >> .gitignore
git add .gitignore
git commit -m "Add .gitignore"
git push [remote repo URL]



2

#create a iris.py file 
#load Iris data using this code : 
from sklearn.datasets import load_iris
import pandas as pd

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df.to_csv("iris.csv", index=False)


#next in terminal:
git init
dvc init
dvc add iris_data.csv
git add iris_data.csv.dvc .gitignore .dvc/
git commit -m "Track dataset with DVC"
git push origin main

3

# Save the script as train_model.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)
print("Model trained")


#in terminal

git add train_model.py
git commit -m "Add model training script"
git push origin main 



4

#create predict_api.py :
from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
app = Flask(__name__)
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True)


#in terminal 
git add predict_api.py
git commit -m "MOdel api"
git push origin main 


5

Prerequisites:
predict_api.py
from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
app = Flask(__name__)
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True)
requirements.txt with Flask and scikit-learn

## create docker file on vs code

# Dockerfile content

FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "predict_api.py"]

# Build and run

docker build -t iris-api .
docker run -p 5000:5000 iris-api


6

##create a file named app.py

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)
st.title("Iris Prediction")

inputs = [st.slider(l, *v) for l, v in zip(
    ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    [(4.0, 8.0, 5.1), (2.0, 4.5, 3.5), (1.0, 7.0, 1.4), (0.1, 2.5, 0.2)])]

if st.button('Predict'):
    st.success(f"Prediction: {iris.target_names[model.predict([inputs])[0]]}")

## to run in terminal

streamlit run app.py


7

## create nginx.conf file 

##paste this


http {
    upstream ml_backend {
        server localhost:5000;
        server localhost:5001;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://ml_backend;
        }
    }
}

## run this in 2 different terminals to get output
# In one terminal
python app.py --port=5000

# In another terminal
python app.py --port=5001


---------------------------------------OR---------------------------

add this code for predict_api.py file:

if _name_ == "_main_":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

##run using these
_in 1 terminal 

set PORT=5000
python predict_api.py

_in 2nd terminal 

set PORT=5001
python predict_api.py



8

# Save as locustfile.py


from locust import HttpUser, task
class MLTest(HttpUser):
    @task
    def predict(self):
        self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})


# in terminal

locust -f locustfile.py --host=http://localhost:5000

9

#add this as bias.py


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in double_scalars")

# Load the Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=column_names, sep=', ', engine='python')

# Prepare data
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Check for bias in original data
print("Original data distribution by sex:")
print(data.groupby(['sex', 'income']).size())

# Create a copy of the sex column before encoding
sex_column = data['sex'].copy()

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 
                                           'occupation', 'relationship', 'race', 
                                           'native-country'])

# Split data
X = data_encoded.drop(['income', 'sex'], axis=1)
y = data_encoded['income']

# Use the original sex column for stratification and analysis
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X, y, sex_column, test_size=0.2, random_state=42, stratify=sex_column)

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Check accuracy by sex
y_pred = model.predict(X_test_scaled)

# Create masks for male and female samples
male_mask = sex_test == ' Male'
female_mask = sex_test == ' Female'

# Calculate accuracy only if there are samples in each group
print("\nBefore mitigation:")
print(f"Overall accuracy: {accuracy_score(y_test, y_pred):.4f}")

if sum(male_mask) > 0:
    male_accuracy = accuracy_score(y_test[male_mask], y_pred[male_mask])
    print(f"Male accuracy: {male_accuracy:.4f}")
else:
    print("No male samples in test set")

if sum(female_mask) > 0:
    female_accuracy = accuracy_score(y_test[female_mask], y_pred[female_mask])
    print(f"Female accuracy: {female_accuracy:.4f}")
else:
    print("No female samples in test set")

# Mitigate bias using reweighting
male_count = sum(sex_train == ' Male')
female_count = sum(sex_train == ' Female')

print(f"\nTraining data distribution: {male_count} males, {female_count} females")

if male_count == 0 or female_count == 0:
    print("Warning: Missing gender samples in training data. Using equal weights.")
    sample_weights = np.ones(len(y_train))
else:
    # Calculate class weights
    male_weight = 1.0
    female_weight = male_count / female_count
    print(f"Using weights: Male={male_weight:.2f}, Female={female_weight:.2f}")
    
    # Apply weights
    sample_weights = np.array([male_weight if sex == ' Male' else female_weight for sex in sex_train])

# Retrain with weights
model_mitigated = LogisticRegression(max_iter=1000)
model_mitigated.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Check accuracy after mitigation
y_pred_mitigated = model_mitigated.predict(X_test_scaled)

print("\nAfter mitigation:")
print(f"Overall accuracy: {accuracy_score(y_test, y_pred_mitigated):.4f}")

if sum(male_mask) > 0:
    male_accuracy = accuracy_score(y_test[male_mask], y_pred_mitigated[male_mask])
    print(f"Male accuracy: {male_accuracy:.4f}")
else:
    print("No male samples in test set")

if sum(female_mask) > 0:
    female_accuracy = accuracy_score(y_test[female_mask], y_pred_mitigated[female_mask])
    print(f"Female accuracy: {female_accuracy:.4f}")
else:
    print("No female samples in test set")



10

#create a file named 
.github , under that , workflows  ,, under that train.yml file 
# .github/workflows/train.yml

== enter this code

name: Train ML Model
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run training
        run: python train_model.py

#next go to github and actions and then run




Sample VIVA



Q: What is the purpose of `.gitignore` in ML projects?
A: To exclude files/folders like virtual environments, compiled files (`*.pyc`), and large datasets from Git tracking.

Q: How does DVC help in versioning large datasets?
A: DVC tracks metadata using `.dvc` files while storing actual data in external storage, enabling reproducible pipelines.

Q: What command initializes both Git and DVC for a new ML project?
A: `git init` followed by `dvc init`.

Q: How do `.dvc` files help in reproducibility?
A: They store file hash and path, allowing restoration of exact data versions via `dvc pull`.

Q: Explain the steps in training a logistic regression model using scikit-learn.
A: Load data → Create model object → Fit model with training data → Save or use the model for prediction.

Q: How can you ensure that a trained model is consistent across runs?
A: Set a fixed `random_state` and control for randomness in training and data splitting.

Q: How do you expose a POST endpoint using Flask?
A: Use `@app.route('/endpoint', methods=['POST'])` and read input via `request.json`.

Q: What are the possible issues if the input JSON format is not validated?
A: It may throw runtime errors like `KeyError`, `TypeError`, or return incorrect predictions.

Q: Explain each instruction in the Dockerfile used for your Flask API.
A: `FROM`: base image, `WORKDIR`: set working directory, `COPY`: add files, `RUN`: install dependencies, `CMD`: run app.

Q: What is the difference between `CMD` and `ENTRYPOINT` in Docker?
A: `CMD` provides default args; `ENTRYPOINT` sets the executable. `CMD` can be overridden easily.

Q: How do you expose a containerized service on port 5000?
A: Use `docker run -p 5000:5000 <image-name>`.

Q: What is the significance of the `WORKDIR` and `COPY` commands?
A: `WORKDIR` sets context for commands; `COPY` transfers local files into the container.

Q: What are the benefits of using Streamlit over Flask for ML model inference?
A: Streamlit offers faster development, built-in widgets, and is better suited for interactive ML UIs.

Q: How would you handle invalid user inputs in Streamlit sliders?
A: Set `min_value`, `max_value`, and `default` in `st.slider()` and validate ranges before prediction.

Q: How does the `upstream` block in NGINX help in balancing traffic?
A: It groups multiple backend servers and balances requests using round-robin or other algorithms.

Q: Why do we use `proxy_pass` and what are its potential pitfalls?
A: `proxy_pass` forwards requests to upstream servers; misconfigurations can cause broken routing or header issues.

Q: What does the `@task` decorator signify in Locust?
A: It marks a function as a user behavior to be executed during load testing.

Q: How would you simulate 100 concurrent users in Locust?
A: Run: `locust -f locustfile.py --users 100 --spawn-rate 10 --host=http://localhost:5000`

Q: How can class imbalance lead to biased ML models?
A: Models overfit to dominant classes, reducing accuracy and fairness for minority groups.

Q: What methods can be used to mitigate bias in datasets?
A: Reweighting, oversampling, undersampling, or fairness-aware algorithms.

Q: How do you automate model training with GitHub Actions?
A: Use `.yml` workflow to install dependencies and run the training script on each code push.

Q: What are the key steps in a `.yml` GitHub Actions pipeline?
A: `checkout` code → `setup-python` → `install dependencies` → `run training script`
