import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

print("Starting Training...")

dataset_path = "dataset"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=2,
    class_mode="sparse",
    subset="training"
)
print(train_data.class_indices)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=2,
    class_mode="sparse",
    subset="validation"
)

base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training Model...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save("agriculture_model.keras")

print("Model Trained and Saved!")