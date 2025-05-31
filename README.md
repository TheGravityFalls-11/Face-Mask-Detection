```
# 😷 Face Mask Detection App

A deep learning web app that detects whether people in an image are wearing a face mask **properly**, **improperly**, or **not at all**.  
Built using TensorFlow, Keras, OpenCV, and Streamlit.

---

## 📂 Folder Structure

facemask/
├── annotations/              # XML annotations for face mask dataset  
├── images/                   # Training images  
├── app.py                    # Streamlit web app  
├── train.py                  # Script to train the model  
├── face_mask_detector.h5     # Trained model  
├── training_plot.png         # Accuracy/loss graph  
├── requirements.txt          # All dependencies  
└── README.md                 # This file

---

## 🚀 How to Run


### 2. Create and activate virtual environment (optional but recommended)

**On Windows:**
python -m venv venv  
venv\Scripts\activate

**On macOS/Linux:**
python3 -m venv venv  
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Train the model (if needed)  
Ensure you have `images/` and `annotations/` folders with Pascal VOC formatted data.

python train.py

### 5. Run the web app
streamlit run app.py

---

## 🧠 Model Details

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Architecture**: MobileNetV2 + custom head with average pooling and dense layers
- **Classes**:
  - with_mask
  - without_mask
  - incorrect_mask *(if trained accordingly)*
- **Input Size**: 224x224 RGB

---

## 📊 Results

The training loss and accuracy over epochs:

(training_plot.png)

---

## 📦 Dependencies

Listed in `requirements.txt`:

- tensorflow  
- opencv-python  
- numpy  
- matplotlib  
- scikit-learn  
- Pillow  
- streamlit  

To install:
pip install -r requirements.txt

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- TensorFlow & Keras for deep learning  
- OpenCV for image processing  
- Streamlit for the web UI  
- LabelImg + Pascal VOC format for annotations
```
