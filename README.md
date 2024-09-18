# Wheat Disease Detection using CNN

This project implements a Convolutional Neural Network (CNN) to detect and classify wheat diseases using TensorFlow and Keras.

## Requirements

```
tensorflow==2.12.0
keras==2.12.0
matplotlib
numpy
scikit-learn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of 2078 images of wheat leaves and heads, categorized into five classes:

1. Brown rust
2. Fusarium Head Blight
3. Healthy
4. Yellow rust
5. Septoria

## Code

### Import necessary libraries

```python
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
```

### Set up constants

```python
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 50
```

### Load and preprocess the dataset

```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "path/to/your/dataset",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
```

### Split the dataset

```python
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size).take(val_size)
test_ds = dataset.skip(train_size + val_size)
```

### Apply data augmentation

```python
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)
```

### Define the model architecture

```python
model = models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

### Train the model

```python
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)
```

### Evaluate the model

```python
scores = model.evaluate(test_ds)
print(f"Test loss: {scores[0]}")
print(f"Test accuracy: {scores[1]}")
```

### Create a prediction function

```python
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
```

### Visualize results

```python
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")
plt.show()
```

## Usage

1. Prepare your dataset and update the path in the `dataset` loading section.
2. Run the code sections in order to train and evaluate the model.
3. Use the `predict` function to classify new wheat images.

## Future Improvements

1. Experiment with different CNN architectures (e.g., ResNet, VGG)
2. Implement transfer learning using pre-trained models
3. Increase dataset size and diversity
4. Develop a web or mobile interface for easy use by farmers

## Contributing

Feel free to open issues or submit pull requests to improve the project.

## License

[MIT License](LICENSE)
