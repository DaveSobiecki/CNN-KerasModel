import os
from shutil import copy2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.metrics as metrics
import seaborn as sns

from keras import Model, Input
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense

from keras_preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


# ========================================================================
#                                 MODEL
# ========================================================================


class PokemonCnnNet:
    @staticmethod
    def build_branch(inputs, output_size, height, width, col, out_name):
        x = Conv2D(filters=64, kernel_size=(4, 4), input_shape=(height, width, col), activation='relu', padding='same')(
            inputs)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(output_size)(x)
        x = Activation(activation='softmax', name=out_name)(x)

        return x

    @staticmethod
    def build_network_one_branch(category_size, img_shape):
        inputs = Input(shape=img_shape)
        first_branch = PokemonCnnNet \
            .build_branch(inputs, category_size, img_shape[0], img_shape[1], img_shape[2], "Type1")

        model = Model(inputs=inputs, outputs=first_branch)
        losses = {"Type1": "categorical_crossentropy"}
        loss_weights = {"Type1": 1.0}

        model.compile(loss=losses,
                      loss_weights=loss_weights,
                      optimizer='adam',
                      metrics=['accuracy',
                               metrics.TruePositives(),
                               metrics.TrueNegatives(),
                               metrics.FalsePositives(),
                               metrics.FalseNegatives()])
        model.summary()
        return model


# ========================================================================
#                                 UTILS
# ========================================================================

def create_folders(source_dataframe):
    if not os.path.exists('images/train/'):
        os.mkdir('images/train/')
    if not os.path.exists('images/test/'):
        os.mkdir('images/test/')
    if not os.path.exists('images/val/'):
        os.mkdir('images/val/')
    if len(os.listdir('images/train/')) == 0:
        for class_ in source_dataframe['Type2'].unique():
            os.mkdir('images/train/' + str(class_) + '/')
            os.mkdir('images/test/' + str(class_) + '/')
            os.mkdir('images/val/' + str(class_) + '/')


def split_copy_data_to_test_and_train(source_dataframe):
    X_train, X_test, y_train, y_test = train_test_split(
        source_dataframe, source_dataframe['Type1'], test_size=0.2, stratify=source_dataframe['Type1'])

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.2, stratify=y_test)

    for image, type_ in zip(X_train['Path'], y_train):
        copy2(image, 'images/train/' + type_)
    for image, type_ in zip(X_test['Path'], y_test):
        copy2(image, 'images/test/' + type_)
    for image, type_ in zip(X_val['Path'], y_val):
        copy2(image, 'images/val/' + type_)


def initialize_image_data_generator():
    datagen = ImageDataGenerator()
    train = datagen.flow_from_directory('images/train/', color_mode='rgba')
    test = datagen.flow_from_directory('images/test/', color_mode='rgba', shuffle=False)
    val = datagen.flow_from_directory('images/val/', color_mode='rgba')

    print(train.class_indices)

    return datagen, train, test, val


def prepare_data():
    dataset = pd.read_csv("pokemon.csv")
    dataset = dataset.sort_values("Name")

    pokemon_dataframe = pd.DataFrame([])

    image_paths = []
    names = []
    for img in os.scandir("images"):
        image_paths.append(f"images/{img.name}")
        names.append(img.name.split('.')[0])
    pokemon_dataframe['Path'] = image_paths
    pokemon_dataframe['Name'] = names
    res = dataset.merge(pokemon_dataframe, left_on='Name', right_on='Name')
    create_folders(res)
    split_copy_data_to_test_and_train(res)

    return initialize_image_data_generator(), len(pokemon_dataframe)


def present_confusion_matrix(data_frame, epoch_number):
    ax = sns.heatmap(data_frame, annot=True, cmap='Blues')
    ax.set_title('Model predictions\n\n')
    ax.set_xlabel('\nPrediction type')
    ax.set_ylabel('Epochs')
    plt.show()

    final_data_row = data_frame.iloc[epoch_number - 1]
    tp = final_data_row['true_positives']
    fp = final_data_row['false_positives']
    fn = final_data_row['false_negatives']
    tn = final_data_row['true_negatives']
    matrix = np.array([[tp, fp],
                       [fn, tn]])
    ax = sns.heatmap(matrix, annot=True, fmt='g')
    ax.set_title('TRUE                FALSE')
    ax.set_ylabel('FALSE                TRUE')
    plt.show()


# ========================================================================
#                                 MAIN
# ========================================================================

def main():
    (data_generator, train_data, test_data, val_data), data_len = prepare_data()
    model = PokemonCnnNet.build_network_one_branch(19, train_data.image_shape)
    model.fit(train_data, epochs=4, verbose=1, validation_data=test_data)
    model.save('TrainedModel')
    model_metrics = pd.DataFrame(model.history.history)

    model_metrics[['loss', 'val_loss']].plot()
    model_metrics[['accuracy', 'val_accuracy']].plot()
    plt.show()

    values_for_cf_matrix = model_metrics[['true_positives', 'true_negatives', 'false_positives', 'false_negatives']]

    present_confusion_matrix(values_for_cf_matrix, 4)


if __name__ == '__main__':
    main()
