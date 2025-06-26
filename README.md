
# Customer Churn Prediction Web Application

This project is a **Customer Churn Prediction** web application built using **Streamlit** and powered by a deep learning model developed with **TensorFlow/Keras**. The app predicts whether a bank customer is likely to churn based on their demographic and financial data, achieving an accuracy of **88%** on the test dataset.

The application provides an intuitive and user-friendly interface for inputting customer details and viewing churn predictions.

---

## 🔧 Features

- **Interactive Interface**: Clean and responsive UI with sidebar input fields and styled output for predictions.
- **Deep Learning Model**: Utilizes a TensorFlow/Keras neural network trained to predict customer churn.
- **Input Parameters**:
  - Credit Score
  - Geography
  - Gender
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Credit Card status
  - Active Member status
  - Estimated Salary
- **Preprocessing**:
  - `LabelEncoder` for Gender
  - `OneHotEncoder` for Geography
  - `StandardScaler` for numerical features
- **Styled Output**: Color-coded prediction (🔴 Red = High Churn, 🟢 Green = Low Churn)

---

## 🚀 Installation & Setup

### ✅ Prerequisites

- Python 3.11
- Streamlit
- TensorFlow
- Pandas
- NumPy
- Scikit-learn

### 📦 Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ParthBharadia/ANN-Classification-Churn.git
   cd ANN-Classification-Churn


2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App Locally**:

   ```bash
   streamlit run app.py
   ```

---

## 🧾 Requirements

Create a `requirements.txt` file with the following:

```
streamlit==1.25.0
tensorflow==2.12.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
```

---

## 💻 Usage

1. Launch the app:

   ```bash
   streamlit run app.py
   ```

2. Access in browser at: [http://localhost:8501](http://localhost:8501)

3. In the **sidebar**, input customer details:

   * **Geography**: Select location
   * **Gender**: Male / Female
   * **Age**: 18 – 92
   * **Tenure**: 0 – 10 years
   * **Credit Score**: 300 – 850
   * **Balance**: Enter account balance
   * **Number of Products**: 1 – 4
   * **Has Credit Card**: Yes (1) / No (0)
   * **Is Active Member**: Yes (1) / No (0)
   * **Estimated Salary**: Enter salary

4. Click **Predict Churn** to get the churn probability and color-coded prediction.

---

## 🧠 Model Details

* **Framework**: TensorFlow / Keras
* **Architecture**: Deep Neural Network
* **Accuracy**: 88% on test dataset
* **Preprocessing**:

  * `LabelEncoder`: Gender
  * `OneHotEncoder`: Geography
  * `StandardScaler`: Numerical features

### 📁 Saved Artifacts:

* `model.h5`: Trained Keras model
* `label_encoder_gender.pkl`: Label encoder for Gender
* `one_hot_encoder_geo.pkl`: One-hot encoder for Geography
* `scaler.pkl`: Feature scaler for input normalization

---

## 🌐 Deployment

The app is deployed on **Streamlit Cloud**.

🔗 **Visit the app**: [App](https://ann-classification-churn-7zpckepf29yseeldazb3q2.streamlit.app/)

---


## 🙏 Acknowledgments

* Built with ❤️ using **Streamlit** for the front-end
* Powered by **TensorFlow/Keras** for the machine learning model
* Inspired by real-world **customer churn prediction** use cases in the banking sector


