🚀 TADC Project

Machine learning project for data analysis and modeling using Python.

📁 Project Structure
.
├── data/                       # Raw and processed datasets
├── docs/                       # Documentation files
├── models/                     # Saved models (trained artifacts)
├── src/                        # Source code (training, preprocessing, etc.)
├── tests/                      # Unit tests
├── TADC_Complete_v3_avec_S.xlsx # Main dataset (Excel format)
├── app.py                      # Main application (Streamlit or script entry point)
├── notebook.ipynb              # Development notebook
├── notebook_executed.ipynb     # Executed notebook with outputs
├── requirements.txt            # Python dependencies
⚙️ Prerequisites
Python 3.10+
pip or virtualenv
🔧 Installation
1. Clone the repository
git clone <your-repo-url>
cd <your-project-folder>
2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3. Install dependencies
pip install -r requirements.txt
▶️ Run the Application
Option 1 — Streamlit app
streamlit run app.py
Option 2 — Python script
python app.py
📊 Data
Main dataset: TADC_Complete_v3_avec_S.xlsx
Place additional datasets inside data/
🧠 Models
Trained models are stored in models/

Typically saved using:

import joblib
joblib.dump(model, "models/model.pkl")
🧪 Testing

Run tests using:

pytest tests/
📓 Notebooks
notebook.ipynb → development / experimentation
notebook_executed.ipynb → final executed version

Run locally with:

jupyter notebook
⚠️ Common Issues
❌ ModuleNotFoundError (e.g. joblib)

Fix:

pip install -r requirements.txt
❌ Streamlit app not starting

Check:

app.py exists at root
All dependencies installed
No missing model files in models/
📦 requirements.txt example

If needed:

pandas
numpy
scikit-learn
xgboost
matplotlib
streamlit
joblib
pytest
openpyxl
shap
🧩 Tips
Keep preprocessing logic inside src/
Avoid putting heavy logic inside app.py
Version your models (model_v1.pkl, etc.)
Use .env if you add secrets later
