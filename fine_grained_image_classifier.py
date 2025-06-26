
# Fine-Grained Image Classification on CIFAR-100
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train, x_val = x_train[:40000], x_train[40000:]
y_train, y_val = y_train[:40000], y_train[40000:]

# Normalize and one-hot encode
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0
y_train_oh = to_categorical(y_train, 100)
y_val_oh = to_categorical(y_val, 100)
y_test_oh = to_categorical(y_test, 100)

# Data augmentation
train_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    zoom_range=0.2
)
val_gen = ImageDataGenerator()

# Custom CNN
def build_custom_cnn():
    model = models.Sequential([
        layers.Conv2D(64, (3,3), padding='same', input_shape=(32,32,3)),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(64, (3,3), padding='same'), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(), layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), padding='same'), layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(128, (3,3), padding='same'), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(), layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256), layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.5),
        layers.Dense(100, activation='softmax')
    ])
    return model

custom_model = build_custom_cnn()
custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

# Train custom CNN
custom_model.fit(
    train_gen.flow(x_train, y_train_oh, batch_size=64),
    epochs=25,
    validation_data=val_gen.flow(x_val, y_val_oh)
)

# Transfer Learning
def build_transfer_model():
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(32,32,3), weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(100, activation='softmax')
    ])
    return model

transfer_model = build_transfer_model()
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

# Stage 1: train classifier head
transfer_model.fit(
    train_gen.flow(x_train, y_train_oh, batch_size=64),
    epochs=5,
    validation_data=val_gen.flow(x_val, y_val_oh)
)

# Stage 2: fine-tune top layers
transfer_model.layers[0].trainable = True
transfer_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
transfer_model.fit(
    train_gen.flow(x_train, y_train_oh, batch_size=64),
    epochs=10,
    validation_data=val_gen.flow(x_val, y_val_oh)
)

# Evaluation
y_pred = np.argmax(custom_model.predict(x_test), axis=1)
print("Custom CNN Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix (matplotlib only)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm[:10, :10], interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Subset of Classes)")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

plt.title("Confusion Matrix (Subset of Classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
