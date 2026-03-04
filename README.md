# AI Phishing Detection Browser Extension

A browser extension that detects phishing websites in real-time using a combination of **machine learning, rule-based analysis, and logo similarity detection**.

The extension analyzes URLs, page features, and website logos to identify suspicious or fraudulent websites before users interact with them.

---

## 🚀 Features

• Real-time phishing detection directly in the browser  
• Machine learning model for URL classification  
• Logo similarity detection using CNN embeddings  
• Suspicious domain pattern detection  
• Lightweight Chrome extension interface  
• Local Flask backend for ML inference  

---

## 🧠 How It Works

1. The browser extension extracts the current website URL and favicon.
2. The URL is analyzed using a trained machine learning model.
3. The favicon is compared with known brand logos using **MobileNetV2 embeddings**.
4. The backend calculates a phishing risk score.
5. The extension displays the risk level to the user.

---

## 🏗 Architecture
User visits website
│
▼
Browser Extension
│
▼
Feature Extraction
(URL + favicon)
│
▼
Flask Backend API
│
├── ML URL Classifier
│
└── Logo Similarity Detection
(MobileNetV2 embeddings)
│
▼
Risk Score Generated
│
▼
Displayed in Extension Popup

---

## 🛠 Tech Stack

### Frontend (Extension)
- JavaScript
- HTML
- CSS
- Chrome Extension APIs

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- TensorFlow / Keras
- MobileNetV2

### Data Processing
- NumPy
- Pandas
- PIL

---

## 📂 Project Structure
phishing-extension
│
├── extension
│   ├── background.js
│   ├── content.js
│   ├── popup.js
│   ├── popup.html
│   ├── popup.css
│   └── manifest.json
│
├── logo_pipeline
│   ├── build_logo_embeddings.py
│   ├── server_logo.py
│   ├── scripts/
│   └── data/
│
├── .gitignore
└── README.md



## ⚙️ Running the Project

### 1. Start the ML backend

```bash
cd logo_pipeline
python server_logo.py

2. Load the extension
	1.	Open Chrome
	2.	Go to chrome://extensions/
	3.	Enable Developer Mode
	4.	Click Load unpacked
	5.	Select the extension folder

The extension will now start analyzing websites.

📌 Future Improvements

• Deploy backend as a cloud API
• Add deep learning webpage content analysis
• Improve phishing dataset coverage
• Add real-time threat intelligence feeds

Author

Shaunak Sardeshpande
Computer Engineering Student | Interested in AI, Cybersecurity, and Startups
