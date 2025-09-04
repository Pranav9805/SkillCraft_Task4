import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_and_preprocess():
    train_df = pd.read_csv('data/sign_mnist_train.csv')
    test_df = pd.read_csv('data/sign_mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_train = to_categorical(y_train, num_classes=25)
    y_test = to_categorical(y_test, num_classes=25)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_preprocess()
    print("Preprocessing complete.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
