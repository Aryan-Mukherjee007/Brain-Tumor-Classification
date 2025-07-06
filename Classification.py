#Imports
import zipfile
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import shutil
import random
import IPython.display as display

#Defining the paths for dataset
zip_path = "/content/archive.zip"
extract_path = "/content/dataset"

#Extracting the zip file archive.zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

train_dir = os.path.join(extract_path, "Training")
test_dir = os.path.join(extract_path, "Testing")

IMG_SIZE = (150, 150)

#Standardizing images to RGB and the same image size
def resize_images_in_directory(directory):
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path): continue
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(IMG_SIZE)
                    img.save(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

#Resizing training and testing set
resize_images_in_directory(train_dir)
resize_images_in_directory(test_dir)

#Generating training and validation batches of size 32 to fit the model
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

#Generating a single batch of testing data (all of it)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

#CNN Model Architecture with 2D convolutions, 3 convolutional layers, and an activation layer
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

#Compiling and summarizing model (for paper)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Running the model with 15 epochs through training set
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

#Predicting on testing set and extracting true labels
test_generator.reset()
preds = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

#Calculating and displaying accurcacy as percentage classified correctly
correct = np.sum(predicted_classes == true_classes)
total = len(true_classes)
accuracy = (correct / total) * 100
print(f"Model Accuracy on Test Set: {accuracy:.2f}%")

#Selecting and annotating 15 random test images (for comparison)
os.makedirs("/content/annotated", exist_ok=True)

random_indices = np.random.choice(total, size=15, replace=False)

for idx, i in enumerate(random_indices):
    img_path = test_generator.filepaths[i]
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    draw = ImageDraw.Draw(img)
    pred_label = class_labels[predicted_classes[i]]
    true_label = class_labels[true_classes[i]]
    draw.text((5, 5), f"Pred: {pred_label}", fill="red")
    draw.text((5, 25), f"True: {true_label}", fill="green")
    out_path = f"/content/annotated/pred_{idx+1}.jpg"
    img.save(out_path)

#Displaying the randomly selected images (for paper)
for i in range(15):
    display.display(Image.open(f"/content/annotated/pred_{i+1}.jpg"))
