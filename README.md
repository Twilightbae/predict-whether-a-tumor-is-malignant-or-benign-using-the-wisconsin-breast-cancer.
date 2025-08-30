🧬 Breast Cancer Detection with Neural Networks + Explainable AI

This project builds a Neural Network classifier for Breast Cancer detection, enhanced with Explainable AI techniques (LIME/SHAP) and a Risk Scoring system.
Finally, an interactive Streamlit dashboard is provided for real-time predictions.

🚀 Project Highlights

✅ Data Preprocessing → Cleaned & standardized breast cancer dataset.

✅ Neural Network Model → Binary classification (Benign vs Malignant).

✅ Risk Scoring System → Outputs a cancer risk percentage (0–100%).

✅ Explainable AI → LIME used to show feature importance for each prediction.

✅ Interactive Dashboard → Built with Streamlit for user-friendly predictions.

📂 Project Structure
breast_cancer_detection_project/
├── notebook.ipynb              # Colab notebook (Phases 1–3: Data → Model → LIME)
├── app.py                      # Streamlit app (Phase 4)
├── nn_breast_cancer_model.h5   # Saved trained model
├── requirements.txt            # Dependencies list
└── README.md                   # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/breast_cancer_detection_project.git
cd breast_cancer_detection_project

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

🧠 Neural Network Architecture

Dense (32, ReLU) → Dropout (0.3)

Dense (16, ReLU) → Dropout (0.2)

Dense (1, Sigmoid) → Output

📊 Results & Evaluation

Accuracy: ~95% (depends on train/test split).

Confusion Matrix: Shows true vs predicted labels.

Risk Index: Patient-wise cancer risk score.

🔍 Explainability with LIME

Provides feature-wise explanations for predictions.

Example: "Mean radius" contributed +0.45 to Malignant prediction.

🌐 Streamlit Dashboard

Upload patient features or enter manually.

Get prediction + risk score.

Visualize feature importance for interpretability.

📌 Future Improvements

🔹 Add SHAP for global + local explanations.

🔹 Deploy on Streamlit Cloud / Hugging Face Spaces.

🔹 Integrate with real-world datasets (medical imaging).

🤝 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📜 License

MIT License

⚡ Author: Ankita
💡 "AI in healthcare should be interpretable and trustworthy."
