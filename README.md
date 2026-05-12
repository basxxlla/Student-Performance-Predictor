# Student-Performance-Predictor
Modern machine learning desktop app for predicting student academic performance using Python and CustomTkinter.

The application provides:

* Performance prediction
* Study habit analysis
* Sensitivity analysis
* Personalized recommendations
* Prediction history tracking
* CSV export functionality

---

##  Features

###  Prediction System

Predicts a student's expected final score based on:

* Study hours
* Attendance
* Previous grades
* Sleep schedule
* Screen time
* Extracurricular activities
* Study group participation

---

###  Analysis Dashboard

Provides:

* Factor contribution analysis
* Sensitivity analysis
* Performance categorization:

  * Excellent
  * Good
  * Average
  * Needs Work

---

###  Smart Recommendations

Generates personalized advice based on user inputs, such as:

* Improving attendance
* Reducing screen time
* Increasing study hours
* Joining study groups
* Improving sleep schedule

---

###  History System

* Stores prediction history locally in JSON format
* Displays previous predictions
* Export history to CSV

---

##  Technologies Used

* Python
* CustomTkinter
* NumPy
* scikit-learn
* Joblib
* JSON & CSV for local storage

---

##  Machine Learning Model

The application automatically:

1. Generates a synthetic student dataset
2. Trains a Linear Regression model
3. Saves the trained model locally
4. Loads the model during application startup

Model Features:

* Study hours
* Attendance
* Previous score
* Sleep hours
* Extracurricular participation
* Screen time
* Study group participation

---

## Project Structure

```bash
student-performance-predictor/
│
├── main.py
├── student_model.pkl
├── label_encoder.pkl
├── label_encoder2.pkl
├── prediction_history.json
├── requirements.txt
└── README.md
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/basxxlla/student-performance-predictor.git
cd student-performance-predictor
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the application

```bash
python main.py
```
---

##  How It Works

The prediction score is calculated using a trained Linear Regression model based on multiple academic and lifestyle factors.

The app then:

* Calculates predicted score
* Assigns a performance category
* Generates recommendations
* Displays analytical insights

---

##  Future Improvements

Possible future enhancements:

* Real dataset integration
* Multiple ML models comparison
* Dark mode support
* User authentication system
* Charts and graphs
* PDF report export
* Database integration
* Student profile system

---

##  Educational Purpose

This project was created for learning purposes and demonstrates:

* GUI development
* Machine learning integration
* Data preprocessing
* Model persistence
* File handling
* Application state management
* User experience design
