# 📄 AI-Powered Math Marking Assistant Documentation

Welcome to the **AI-Powered Math Marking Assistant**! This app evaluates handwritten math solutions using OCR (Optical Character Recognition) and AI from OpenAI.

---

## 🚀 How to Download and Run the App

### ✅ Step 1: Prerequisites

Make sure you have the following installed on your computer:

1. **Docker** (Recommended for easy setup)  
   - [Download Docker](https://www.docker.com/products/docker-desktop)
   - Verify Docker installation:
     ```bash
     docker --version
     ```

   **OR**

   **Python** (Version 3.7 or later) if not using Docker:
   - Check Python version:
     ```bash
     python --version
     ```
   - If not installed, [download Python](https://www.python.org/downloads/).

2. **Git** (to download from GitHub):
   - Verify Git installation:
     ```bash
     git --version
     ```
   - If not installed, [download Git](https://git-scm.com/downloads).

---

### 📥 Step 2: Download the Project from GitHub

1. Open your terminal (Command Prompt, PowerShell, or Terminal on macOS/Linux).
2. Clone the GitHub repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
   *(Replace `your-username/your-repo-name` with the actual GitHub repository URL.)*

3. Move into the project folder:
   ```bash
   cd your-repo-name
   ```

---

### ⚙️ Step 3: Run the App

#### **Option A: Using Docker (Recommended)**

1. Build the Docker image:
   ```bash
   docker build -t math-marking-assistant .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 math-marking-assistant
   ```

This will start the app at `http://localhost:8501`.

#### **Option B: Running Locally with Python**

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

The app will open in your web browser automatically. If not, go to `http://localhost:8501`.

---

### 🔑 Step 4: Using the App

1. **Enter Your OpenAI API Key:**
   - In the sidebar, enter your **OpenAI API Key**.
   - Don’t have one? Sign up at [OpenAI](https://platform.openai.com/signup).

2. **Upload Your Handwritten Math Solution:**
   - Click **"Choose an image file"** to upload your handwritten solution (JPEG, PNG, etc.).

3. **Evaluate the Solution:**
   - Click **"Evaluate Solution"**.
   - You'll receive:
     - ✅ **Score** (out of 10)
     - 💬 **Feedback** on correctness
     - 📊 **Token Usage** (to track API consumption)

---

### 🛠️ Troubleshooting

- **Python Not Recognized:**
  - Ensure Python is added to your system’s PATH during installation.

- **Docker Build Errors:**
  - Make sure Docker is running and that you have network access.

- **API Key Errors:**
  - Double-check your OpenAI API key. Ensure it’s active and has usage limits available.

- **Port Error (App Already Running):**
  - Stop previous instances with `Ctrl + C` in the terminal.
  - Or run on a new port:
    ```bash
    streamlit run app.py --server.port 8502
    ```

---

### 🤝 Contributing

Want to improve the app? Fork the repo, make changes, and submit a pull request!

---

### 📧 Support

Need help? Open an issue on the GitHub repo or contact the project maintainer.

---

