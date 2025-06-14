# 💎 Gemstone Price Prediction

This project predicts the price of gemstones based on various parameters using a Machine Learning model (CatBoost) integrated into a Flask web application.

## 🚀 Project Overview

The Gemstone Price Prediction app allows users to input various characteristics of a gemstone such as:

- Carat
- Depth
- Table
- x, y, z dimensions
- Cut quality
- Color grade
- Clarity grade

Based on these inputs, the model predicts the price of the gemstone.

The web interface is built using **Flask** and styled with HTML, CSS, and basic frontend frameworks.

---

## 🧰 Tech Stack

- **Python**
- **Flask (Backend Framework)**
- **CatBoost (ML Model)**
- **HTML/CSS/JavaScript (Frontend)**
- **Pandas, NumPy, Scikit-learn (Data Processing)**
- **Git & GitHub (Version Control)**

---

## 🖼 Demo Screenshot

Here is a screenshot of the working web application interface:

![Gemstone Price Prediction UI](static\css\image\image.png)


## 📂 Project Structure


├── .ebextensions/ # Elastic Beanstalk configs (if deploying)

├── .vscode/ # VS Code configs

├── artifacts/ # Contains trained model and artifacts

├── catboost_info/ # CatBoost model info directory

├── notebook/ # Jupyter Notebooks for data 
ingestion & EDA

├── src/ # Source code for ML pipeline

├── static/css/ # Styling files for frontend

├── templates/ # HTML templates (Jinja2 for Flask)

├── .gitignore

├── README.md

├── application.py # Flask application entry point

├── requirements.txt # Python dependencies

├── setup.py # Package setup



---

## 📊 Model Information

- **Algorithm Used**: CatBoost Regressor
- **Target Variable**: Price
- **Input Features**:
  - carat
  - depth
  - table
  - x, y, z
  - cut
  - color
  - clarity

---

## 🖥 Web Interface

The user-friendly web interface takes gemstone properties as input and displays the predicted price after model inference.

Example form:

- Carat: 3.5  
- Depth: 5  
- Table: 5  
- x: 3.1  
- y: 3.4  
- z: 2.3  
- Cut: Fair  
- Color: D  
- Clarity: I1  

Output: `Predicted Gemstone Price`

---

## ⚙ How to Run Locally

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/kanishka-rani-2005/GemPricePredictor.git
cd GemPricePredictor
