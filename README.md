# Visual Product Matcher

A web application that allows users to find visually similar products by uploading an image or providing an image URL.

## Features
- Upload image via file or URL
- MobileNetV2-based feature extraction
- Visual similarity search with cosine similarity
- Filter results by similarity score
- Displays product metadata with images
- Responsive and clean UI using Bootstrap and custom styles

## Technologies
- Python, Flask
- TensorFlow (MobileNetV2)
- Pandas, NumPy, scikit-learn
- Bootstrap CSS for frontend

## Setup and Installation

1.git clone <repo_url> ## Download the project repository to local machine
2.cd <repo_folder> # Change directory to the project folder
3.python -m venv venv  # Create a virtual environment named 'venv' for dependency isolation
4.source venv/bin/activate  # Activate the virtual environment 
5.pip install -r requirements.txt # Install all required Python packages listed in requirements.txt
6.python app.py    # Run the Flask web application locally


