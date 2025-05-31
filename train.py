import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
IMG_SIZE = 224

# Prepare data
print("[INFO] Loading data...")
data = []
labels = []

for xml_file in os.listdir(ANNOTATION_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, filename)

    image = cv2.imread(image_path)
    if image is None:
        continue

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        x1 = int(bbox.find("xmin").text)
        y1 = int(bbox.find("ymin").text)
        x2 = int(bbox.find("xmax").text)
        y2 = int(bbox.find("ymax").text)

        face = image[y1:y2, x1:x2]
        try:
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        except Exception as e:
            print(f"Skipping face due to resize error: {e}")
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = preprocess_input(face)
        data.append(face)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_cat = to_categorical(labels_encoded)

print(f"[INFO] Classes found: {le.classes_}")

# Split data
(trainX, testX, trainY, testY) = train_test_split(
    data, labels_cat, test_size=0.2, stratify=labels_encoded, random_state=42
)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Build model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(le.classes_), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile
print("[INFO] Compiling model...")
opt = Adam(learning_rate=1e-4, decay=1e-4 / 5)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train
print("[INFO] Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=8),
    steps_per_epoch=len(trainX) // 8,
    validation_data=(testX, testY),
    validation_steps=len(testX) // 8,
    epochs=5
)

# Save model
print("[INFO] Saving model to face_mask_detector.h5...")
model.save("face_mask_detector.h5")

# Plot training loss and accuracy
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
