import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
import matplotlib.pyplot as plt

original_dataset_dir = "gestures_dataset"
base_dir = "processed_gestures_dataset"
os.makedirs(base_dir, exist_ok=True)

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

categories = ["down", "level", "low", "ok", "stop", "up"]

for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, category), exist_ok=True)

for category in categories:
    category_dir = os.path.join(original_dataset_dir, category)
    images = [img for img in os.listdir(category_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    for img_path in train_images:
        shutil.copy(os.path.join(category_dir, img_path), os.path.join(train_dir, category, img_path))
    for img_path in val_images:
        shutil.copy(os.path.join(category_dir, img_path), os.path.join(validation_dir, category, img_path))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

def create_gesture_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape), #wypełnienie
        BatchNormalization(),#bez znikająchych gradientów , na rgb
        MaxPooling2D(2, 2),

        Conv2D(64, (5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),


        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_gesture_model(
    input_shape=(150, 150, 3),
    num_classes=train_generator.num_classes
)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_reducer = ReduceLROnPlateau( #slow ucz jak strat walid nie popraw
    monitor='val_loss',
    factor=0.5,
    patience=5
)

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[early_stop, lr_reducer]
)

model.save("gesture_classification_model.keras")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Gesture Classification)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (Gesture Classification)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("saved_plot.png")
print("Plot saved as 'saved_plot.png'.")


