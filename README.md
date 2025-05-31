😷 Face Mask Detection App
A deep learning web app that detects whether people in an image are wearing a face mask properly, improperly, or not at all. Built using TensorFlow, Keras, OpenCV, and Streamlit.

📂 Folder Structure
bash
Copy
Edit
facemask/
├── annotations/              # XML annotations for face mask dataset
├── images/                   # Training images
├── app.py                    # Streamlit web app
├── train.py                  # Script to train the model
├── face_mask_detector.h5     # Trained model
├── training_plot.png         # Accuracy/loss graph
├── requirements.txt          # All dependencies
└── README.md                 # This file
🚀 How to Run
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/facemask.git
cd facemask
2. Create and activate virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Train the model (if needed)
Ensure you have images/ and annotations/ with proper data in Pascal VOC format.

bash
Copy
Edit
python train.py
5. Run the web app
bash
Copy
Edit
streamlit run app.py
🧠 Model Details
Base model: MobileNetV2 (pre-trained on ImageNet)

Architecture: MobileNetV2 + custom head with average pooling and dense layers

Classes: with_mask, without_mask, incorrect_mask (if trained accordingly)

Input Size: 224x224 RGB

📊 Results
The training loss and accuracy over epochs:



📦 Dependencies
Listed in requirements.txt. Includes:

txt
Copy
Edit
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
Pillow
streamlit
Install them with:

bash
Copy
Edit
pip install -r requirements.txt
📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
TensorFlow & Keras for model building

OpenCV for face detection

Streamlit for frontend
