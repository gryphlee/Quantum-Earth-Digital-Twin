🌍 Quantum Earth Digital Twin v2.3

An integrated simulation suite for predictive climate analytics, quantum optimization, and AI scenario generation.

This project is a fully interactive web application that simulates various environmental data points on a 3D digital globe. It utilizes a hybrid of classical machine learning models, a true quantum circuit simulation (via Qiskit), and advanced visualization techniques to provide insights into future climate scenarios.

✨ Features

The application is divided into several key modules:

🛰️ Digital Twin Visualization

Interactive 3D Globe: Powered by Three.js, featuring full zoom and rotation controls.

Multi-Layer Data Display: Capable of visualizing distinct data layers:

Risk Assessment: Displays 'Normal', 'Warning', and 'Danger' status for cities (Green/Yellow/Red).

Energy Twin: Shows data for 'Solar Potential', 'Flood Risk', and 'Agricultural Viability' using color intensity.

Clickable Data Points: Reveals detailed information upon clicking a city marker on the globe.

🤖 AI & Quantum Models

Temperature Forecasting: Employs LinearRegression (Scikit-learn) to predict future temperature trends.

CO₂ Level Forecasting: Uses a more advanced RandomForestRegressor (Scikit-learn) for CO₂ predictions.

True Quantum Risk Assessment: Leverages Qiskit to run a genuine quantum circuit on an AerSimulator. The risk level is based on the measurement outcome of a qubit whose state is manipulated by the temperature data.

Anomaly Detection: Automatically compares the forecast against a simulated "historical average" to calculate the Temperature Anomaly.

⚡ Advanced Simulation Modes

Quantum Optimization: Recommends the optimal location for a solar farm based on the city with the most "Normal" days over the simulation period.

Global Catastrophe Mode: Simulates a worst-case scenario where global temperatures suddenly spike, demonstrating the immediate impact on risk and other data points.

AI "LLM" Scenario Generator: Automatically generates a future narrative based on all simulated data, including the anomaly score and CO₂ levels.

🛠️ Tech Stack

Backend & Web App: Python, Streamlit

Machine Learning: Scikit-learn (LinearRegression, RandomForestRegressor)

Quantum Computing: IBM Qiskit (QuantumCircuit, AerSimulator)

3D Visualization: Three.js

Data Handling: Pandas, NumPy

📂 Project Structure

QuantumEarthProject/
│
├── .gitignore
├── README.md
│
└── backend/
    ├── venv/                 <-- Your Python virtual environment
    └── app.py                <-- The single source file for the entire Streamlit application


🚀 Setup and Installation

To run this project on your local machine, follow these steps:

Clone the Repository (or download the files).

Set up the Python Virtual Environment:

Open a terminal inside the backend folder.

Run: python -m venv venv

Activate the Environment:

Windows (PowerShell/CMD): .\venv\Scripts\activate

macOS/Linux: source venv/bin/activate

Install the Required Libraries:

Run this command within the activated environment:

pip install streamlit pandas numpy scikit-learn qiskit qiskit-aer


▶️ How to Run

Ensure your virtual environment (venv) is activated.

In your terminal, make sure you are in the backend directory.

Run the command:

streamlit run app.py


Your browser will automatically open with the application running.

🏛️ Architectural Overview

The app employs a simple yet powerful architecture:

All-in-One Backend: Streamlit serves as both the web server and the application's "brain". All data generation, AI model training, and quantum simulations occur in Python.

Frontend-in-Backend: The Three.js code for the 3D globe is embedded within the Python script as a multi-line string.

Data Flow:

The user interacts with controls (sliders, toggles) in the Streamlit sidebar.

Upon clicking "Run New Simulation," the Python backend executes the AI and Quantum models to generate a Pandas DataFrame.

This DataFrame is converted into a JSON object.

The JSON object is passed into the Three.js code (via string.replace()) before being rendered by streamlit.components.v1.html.

The JavaScript code in the browser is responsible for drawing the data points on the 3D globe based on the received JSON data.

🧠 Core Components Explained

run_qiskit_risk_assessment(temperature):
This is the heart of the quantum simulation. It converts the input temperature into a rotation angle (theta). It then builds a quantum circuit with a single qubit, applies an RY(theta) gate, and measures the result. The probability of measuring |1⟩ (Danger) is directly dependent on theta, demonstrating a true quantum probabilistic outcome.

generate_climate_story(...):
This is a "simulated" Large Language Model. It uses Python's f-strings to construct a dynamic narrative based on key metrics from the simulation (e.g., temp_anomaly, avg_co2, recommended_city).

st.session_state:
This is a critical Streamlit feature used to persist the results of the last simulation. It is why the data on the globe and charts does not disappear when other UI elements are manipulated.

