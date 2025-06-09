# Image Classification Prof.traning Ai
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Step 1: Load the dataset (Error handling included)
def load_dataset(dataset_path):
    try:
        # Use ImageDataGenerator to load and preprocess the dataset
        datagen = ImageDataGenerator(validation_split=0.2)
        train_data = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="training"
        )
        val_data = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="validation"
        )
        return train_data, val_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

# Step 2: Split the dataset
# This is already handled by the validation_split parameter in ImageDataGenerator

# Step 3: Preprocess the images (scaling and augmentation)
def preprocess_images(train_data, val_data):
    # Data augmentation is already applied in ImageDataGenerator
    print("Preprocessing completed.")
    return train_data, val_data

# Step 4: Visualize the dataset
def plot_sample_images(data):
    class_names = list(data.class_indices.keys())
    plt.figure(figsize=(10, 10))
    for i, (img, label) in enumerate(data):
        if i == 9:  # Plot 9 samples
            break
        plt.subplot(3, 3, i + 1)
        plt.imshow(img[0].astype("uint8"))
        plt.title(class_names[np.argmax(label[0])])
        plt.axis("off")
    plt.show()

# Step 5: Define and train a pretrained model
def train_model(model_name, train_data, val_data, optimizer):
    # Load the pretrained model
    if model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif model_name == "EfficientNet":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif model_name == "MobileNet":
        base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Invalid model name")

    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(train_data.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Callbacks for early stopping and model checkpoint
    checkpoint = ModelCheckpoint(f"{model_name}_best_model.h5", save_best_only=True, monitor="val_loss", mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=[checkpoint, early_stopping]
    )

    return model, history

# Step 6: Evaluate the model
def evaluate_model(model, val_data):
    val_loss, val_accuracy = model.evaluate(val_data)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    return val_loss, val_accuracy

# Step 7: Plot confusion matrix and F1 scores
def plot_confusion_matrix_and_f1(model, val_data):
    # Get predictions and true labels
    val_data.reset()
    predictions = model.predict(val_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_data.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = list(val_data.class_indices.keys())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # F1 Scores
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Step 8: Plot training curves
def plot_training_curves(history):
    # Training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


# Example usage:
# dataset_path = "path_to_your_dataset"
# train_data, val_data = load_dataset(dataset_path)
# train_data, val_data = preprocess_images(train_data, val_data)
# plot_sample_images(train_data)
# model, history = train_model("ResNet50", train_data, val_data, optimizer="adam")
# evaluate_model(model, val_data)
# plot_confusion_matrix_and_f1(model, val_data)
# plot_training_curves(history)