# Clone the repo
git clone https://github.com/your-username/simple-mood-recognizer.git
cd simple-mood-recognizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install opencv-python tensorflow numpy
pip freeze > requirements.txt
