import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

class DogsCats:
    CLASS_NAMES = ['cat', 'dog']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('/Users/aishwaryadekhane/Desktop/My_Files/Sem-3/PRNN/HW10/dogs-vs-cats')
    SRC_DIR = pathlib.Path('/Users/aishwaryadekhane/Desktop/My_Files/Sem-3/PRNN/HW10/dogs-vs-cats-original/train')
    
    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self, subset_name: str, start_index: int, end_index: int):
        """Create dataset folders by copying images to train/valid/test directories."""
        dest_dir = self.BASE_DIR / subset_name
        os.makedirs(dest_dir, exist_ok=True)

        for category in self.CLASS_NAMES:
            dir_path = dest_dir / category
            dir_path.mkdir(parents=True, exist_ok=True)

            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            for i, file_name in enumerate(files):
                src_path = self.SRC_DIR / file_name
                dst_path = dir_path / file_name

                if src_path.exists():
                    shutil.copyfile(src=src_path, dst=dst_path)
                    if i % 100 == 0:
                        print(f'Copied: {src_path} -> {dst_path}')
                else:
                    print(f'Warning: Source file {src_path} does not exist.')

    def _make_dataset(self, subset_name: str) -> tf.data.Dataset:
        """Create a dataset from a directory."""
        subset_dir = str(self.BASE_DIR / subset_name)
        return tf.keras.utils.image_dataset_from_directory(
            subset_dir,
            image_size=self.IMAGE_SHAPE[:2],
            batch_size=self.BATCH_SIZE
        )

    def make_dataset(self):
        """Load train, validation, and test datasets."""
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('valid')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation: bool = True):
        """Build and compile the CNN model."""
        model = models.Sequential()

        # Data augmentation
        if augmentation:
            model.add(layers.RandomFlip("horizontal"))
            model.add(layers.RandomRotation(0.1))

        # Base model architecture
        model.add(layers.Rescaling(1.0 / 255, input_shape=self.IMAGE_SHAPE))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Fully connected layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Explicitly build the model
        model.build(input_shape=(None, *self.IMAGE_SHAPE))
        self.model = model

        # Print model summary
        self.model.summary()

    def train(self, model_name: str):
        """Train the model and save the best one."""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_name,  # Save in native Keras format (.keras)
                save_best_only=True,
                monitor='val_accuracy',  # Monitor validation accuracy
                mode='max'  # Save when val_accuracy is maximized
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)  # Early stop on validation loss
        ]

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=20,
            callbacks=callbacks
        )

        # Plot training history
        self.plot_history(history)


    def plot_history(self, history):
        """Plot training and validation metrics."""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.show()

    def load_model(self, model_name: str):
        """Load a pre-trained model."""
        self.model = tf.keras.models.load_model(model_name)
        return self.model

    def predict(self, image_file: str):
        """Predict the class of a single image."""
        img = tf.keras.utils.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

        predictions = self.model.predict(img_array)
        predicted_class = self.CLASS_NAMES[int(predictions[0] > 0.5)]

        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class}")
        plt.axis('off')
        plt.show()
