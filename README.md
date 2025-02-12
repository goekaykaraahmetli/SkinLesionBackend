# **Project Setup Guide**

## **Prerequisites**

- **Python Version**: Python 3.10 (Required for TensorFlow compatibility on macOS Apple Silicon)
- **Development Tools**: Terminal, Python, pip

---

## **Setup Instructions**

### **1. Install Python 3.10**

To ensure compatibility with TensorFlow, you must use Python **3.10**.

1. **Install Python 3.10** using Homebrew:
   
bash
   brew install python@3.10


2. **Create a virtual environment** (Required only on macOS, as Python no longer allows direct package installation on macOS):
   
bash
   python3.10 -m venv venv


3. **Activate the virtual environment**:
   
bash
   source venv/bin/activate


4. **Check Python version** to ensure you are using Python 3.10:
   
bash
   python --version

   You should see:
   
Python 3.10.x


---

### **2. Install Necessary Modules**

#### **Option 1: For Systems Without macOS Apple Silicon (Intel Macs)**
Run the following command to install required libraries:
bash
pip install flask tensorflow numpy pillow


#### **Option 2: For macOS with Apple Silicon (M1, M2, M3, M4 Chips)**
For Apple Silicon devices, the **tensorflow-macos** package is required instead of the standard TensorFlow package. Run the following commands:
bash
pip install tensorflow-macos
pip install flask numpy pillow

These commands will install the following dependencies:
- **TensorFlow** (Apple Silicon compatible)
- **Flask** (Web framework for API development)
- **NumPy** (Mathematical computing library)
- **Pillow** (Image processing library)

**Note:** If you encounter issues with pip, ensure you have activated the virtual environment and are using the correct version of Python 3.10.

---

### **3. Start the Backend**

After all the necessary modules are installed, you can start the backend server.

1. Run the backend script using the following command:
   
bash
   python backend.py


2. You should see output similar to this:
   
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


This means the server is successfully running and listening for incoming API requests.

---

## **Usage Instructions**

### **API Endpoint**
To make API calls, use the following endpoint:

http://127.0.0.1:5000/predict


### **How to Call the API**

1. **Method**: POST
2. **URL**: http://127.0.0.1:5000/predict
3. **Body**: Include a **.jpg file** in the **Body** of the request.

You can use tools like **Postman**, **Insomnia**, or **cURL** to send requests.

#### **Example cURL Request**
If you have a .jpg image named image.jpg, you can send the file using the following cURL command:

bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"


Make sure to replace /path/to/your/image.jpg with the actual path to the image on your system.

---

## **Troubleshooting**

### **Issue: TensorFlow is not installed**
**Solution:** Make sure you are using Python 3.10 and have installed tensorflow-macos as described in the setup instructions.

### **Issue: ModuleNotFoundError: No module named 'flask'**
**Solution:** Activate your virtual environment and ensure Flask is installed by running:
bash
source venv/bin/activate
pip install flask


### **Issue: Python version mismatch**
**Solution:** Ensure you are using Python 3.10. Run the following command to verify:
bash
python --version

If it's not 3.10, create a new virtual environment using Python 3.10 as described in the **Setup Instructions**.

---

## **File Structure**
project-folder/
├── backend.py           # The main backend script
├── venv/                # Virtual environment (ignored by Git)
├── .gitignore           # Git ignore file
└── README.md            # This setup guide


---

## **Git Ignore Configuration**
To avoid committing the **venv** folder to GitHub, you should add a .gitignore file in the root of your project with the following content:

# Ignore virtual environments
venv/

# Ignore macOS system files
.DS_Store

# Ignore Python cache files
__pycache__/
*.pyc
*.pyo

# Ignore IDE configuration files
.idea/
.vscode/
