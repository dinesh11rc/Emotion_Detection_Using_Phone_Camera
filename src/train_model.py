import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# Dataset folders
train_dir = 'data/archive/train'
val_dir = 'data/archive/test'

# Check if folders exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise Exception("❌ Dataset folders not found! Make sure 'data/archive/train' and 'data/archive/test' exist.")

# Data Generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
).flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ✅ Print class indices
print("📌 Class indices (emotion-label order):", train_gen.class_indices)

# Load base model
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

# Create final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(7, activation='softmax')  # Update if you have a different number of classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create model save folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('model/emotion_model.keras', save_best_only=True)
]

print("🚀 Training started...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Training done!")
model.save('model/emotion_model.keras')
