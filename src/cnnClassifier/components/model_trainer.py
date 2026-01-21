import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import dagshub

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale = 1./255)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "valid"),
            shuffle=False,
            **dataflow_kwargs
        )
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40, horizontal_flip=True, width_shift_range=0.2,
                height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "train"),
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        dagshub.init(repo_owner='diyaavirmani', repo_name='MLOPs-Production-Ready-Deep-Learning-Project', mlflow=True)
        mlflow.tensorflow.autolog()

        with mlflow.start_run():
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                callbacks=[early_stopping]
            )

            best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
            mlflow.log_metric("best_epoch", best_epoch)
            
            # Local Plotting for quick check
            pd.DataFrame(history.history).plot(figsize=(8,5))
            plt.grid(True)
            plt.savefig("progress_plot.png")
            mlflow.log_artifact("progress_plot.png")
            
            print(f"Best Epoch was: {best_epoch}")
            self.save_model(path=self.config.trained_model_path, model=self.model)
