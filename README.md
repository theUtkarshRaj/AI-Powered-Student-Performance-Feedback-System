# 🎓 AI-Powered Student Performance Feedback System

A Streamlit-based application that analyzes student test performance, visualizes learning trends, and uses the **Google Gemini API** to generate personalized academic feedback. It outputs professional, downloadable reports to help students and educators track and improve learning outcomes.

---

## ✨ Key Features

- **🤖 AI-Generated Feedback**  
  Leverages the Google Gemini API to provide detailed, constructive, and motivational feedback—like having a virtual academic coach.

- **📊 In-Depth Data Analysis**  
  Processes JSON test data to compute:
  - Overall performance (score, accuracy, time utilization)
  - Subject-wise analysis
  - Chapter-level strengths and weaknesses
  - Concept-level mastery insights

- **📈 Rich Visualizations**  
  - Subject Performance Radar Chart  
  - Time vs. Accuracy Scatter Plot  
  - Chapter vs. Difficulty Heatmap  

- **📄 Professional Report Generation**  
  - **Primary:** Styled PDF via ReportLab  
  - **Fallback:** Matplotlib-generated PDF  
  - **Final fallback:** Simple `.txt` report if PDF libraries are unavailable

- **🖥️ Interactive UI**  
  Built with Streamlit: clean layout, metric cards, visual explorer, and AI feedback section.

- **📂 Flexible Data Input**  
  - Upload custom JSON test data  
  - Or use the built-in `data.json` for demo purposes

---

## 📹 Video Demo

Watch a short demo of the app in action:

🎥 [Click here to watch the demo](https://drive.google.com/file/d/1VzmToJVXdNBRl4zG1Th2cNt6fpDLwzQL/view?usp=drive_link)

---

## 🖼️ Screenshots

Here are some previews of the application:

### 📊 Dashboard Overview
![Dashboard Screenshot](Images/Performance_data.png)

### 📈 Visualization
![Radar Chart](Images/Visualization.png)

### 📄 Performance Table
![PDF Report](Images/Performance_data.png)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### 5. Prepare Sample Data

Ensure a `data.json` file is present in the root directory to demo the system.

---

## 🚀 Running the Application

```bash
streamlit run your_script_name.py
```

---

## 📁 File Structure

```
.
├── your_script_name.py     # Main Streamlit application
├── data.json               # Sample student test data
├── .env                    # API key (not included in git)
├── requirements.txt        # Project dependencies
├── Images/                 # Folder for demo images
│   ├── Performance_data.png
│   ├── Visualization.png
└── README.md               # Project documentation
```

---

## 📜 License

```text
MIT License
```

---

## 🌟 Credits & Acknowledgements

* Developed using [Streamlit](https://streamlit.io/)
* Feedback generation powered by [Google Gemini API](https://ai.google.dev/)
* Visualizations via [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
