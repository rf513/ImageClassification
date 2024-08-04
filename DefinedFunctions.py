# import
import os.path
from random import randint
import cv2
import keras
from keras import Sequential
from keras.src.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# function to convert categorical labels (str) to numerical labels
def categ_num_cov(root_path):
    # list of the categories to be classified into (subfolder names)
    categ = os.listdir(root_path)

    # convert categorical labels to numerical labels
    categ_num_dict = dict.fromkeys(categ, )
    for n in range(len(categ)):
        category = categ[n]
        categ_num_dict[category] = n

    # displays label:number conversion list
    return categ_num_dict


# function to create a pandas dataframe (table) listing image directories, categorical label, and numerical label
def img_df(root_path):
    # list of the categories to be classified into (subfolder names)
    categ = os.listdir(root_path)

    # convert categorical labels to numerical labels
    categ_num_dict = categ_num_cov(root_path)

    img_dir_list = []
    label_list = []
    for label in categ:
        # root_path + label points the directory inside the corresponding subfolder
        for img in os.listdir(root_path + label):
            img_dir_list.append(root_path + label + '/' + img)
            label_list.append(label)
    df = pd.DataFrame({'img dir': img_dir_list, 'label': label_list})

    # add a column to the dataframe with corresponding numerical labels
    df['label_num'] = df['label'].map(categ_num_dict)

    return df


# function to display a plot of all the input images under corresponding labels
# the number of images can be reduced for larger datasets
def show_dataset(root_path):
    # list of the categories to be classified into (subfolder names)
    categ = os.listdir(root_path)
    categ_count = len(categ)

    # import the input dataframe
    df = img_df(root_path)

    # create a plt.figure with the number of rows corresponding to the number of categories to be classified
    main = plt.figure(constrained_layout=True, figsize=(16, 8))
    title = os.path.basename(os.path.dirname(root_path))
    main.suptitle(title)
    sub = main.subfigures(categ_count, 1)

    for i, label in enumerate(categ):
        # obtain the number of images there are in each subfolder
        count = df['label_num'].value_counts()[i]

        """ Manually define the number of images to be displayed
        count = (fixed value)
        """

        # calculate the number of rows needed for the images to be displayed in 10 columns
        nrows = (count + 9) // 10

        # create a subplot for each category label
        sub[i].suptitle(label)
        axes = sub[i].subplots(nrows, 10)

        # for cases with 1 row, enclose the axes in an array to form a 2D array for a consistent indexing
        if nrows == 1:
            axes = [axes]

        # display an image in each cell
        for n in range(count):
            # point at the position of the subplot
            ax = axes[n // 10][n % 10]
            # obtain the image directory path
            img_path = df[df['label'] == label]['img dir'].iloc[n]
            ax.imshow(plt.imread(img_path))
            ax.axis('off')

        # for subplots without an image displayed, set the axis off
        for n in range(count, nrows * 10):
            ax = axes[n // 10][n % 10]
            ax.axis('off')
    plt.show()


# function to extract array X (images in numerical array format) and array Y (corresponding labels)
def x_y_dataset(df):
    # shuffle the order of the images in the dataset
    df = shuffle(df)

    # resize and load the images in numerical array format
    x = []
    for img in df['img dir']:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (256, 256))
        img = img / 255
        x.append(img)

    # load all corresponding labels as an array
    y = df['label_num']

    # reformat both X and Y sets to np.array that can be loaded into the model training function
    x = np.array(x)
    y = np.array(y)
    return x, y


# function to check the image shape before and after resizing
def test_image_shape(image_df, x):
    num = randint(0, (len(x) - 1))
    print(f"Array shape of the image before: {plt.imread(image_df['img dir'][num]).shape}")
    print(f"Array shape of the image after: {x[num].shape}")
    if x[num].shape == (256, 256, 3):
        print("The image has been resized correctly\n")
    else:
        print("Error with resizing\n")


# function to check the x,y arrays contain the same number of elements (images and corresponding labels)
def test_dataset_size(x, y):
    if len(x) == len(y):
        print("Categories are split correctly\n")
    else:
        print("Error in dataset split\n")


# function to train a fixed model
def train_model(x_training, y_training, x_validation, y_validation):
    class_count = len(np.unique(y_training))

    # define the layers of the cnn model
    cnn_model = Sequential([
        Input(shape=(256, 256, 3)),
        Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(units=32, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(units=class_count, activation='softmax')
    ])

    # adjust the compilation options
    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # train the model
    training_history = cnn_model.fit(
        x_training,
        y_training,
        batch_size=32,
        epochs=15,
        validation_data=(x_validation, y_validation)
    )

    return training_history, cnn_model


# function to graph the accuracy and loss of the training model
def graph_training(training_history):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.plot(training_history.history['loss'], label='Training', marker='o')
    plt.plot(training_history.history['val_loss'], label='Validation', marker='o')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(training_history.history['accuracy'], label='Training', marker='o')
    plt.plot(training_history.history['val_accuracy'], label='Validation', marker='o')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()

# function to evaluate the model and provide
# df(input information), evaluation accuracy/loss, comparison of prediction vs true label
def model_eval(eval_test_path, eval_model):
    # list of the categories to be classified into (subfolder names)
    categ = os.listdir(eval_test_path)

    # show the categ_num conversion dict
    categ_num_dict = categ_num_cov(eval_test_path)
    print(f"\nCategories and the corresponding numerical label: \n{categ_num_dict}\n")

    # define and show df_train
    df_test = img_df(eval_test_path)
    print(f"Table of files and labels: \n{df_test}\n")

    # visualise dataset from the given path
    show_dataset(eval_test_path)
    x_test, y_test = x_y_dataset(df_test)

    # check the image shape before and after resizing
    test_image_shape(df_test, x_test)
    test_dataset_size(x_test, y_test)

    # making prediction
    print("Test set evaluation")
    eval_model.evaluate(x_test, y_test)
    print("\nTest set model prediction")
    predict_x = eval_model.predict(x_test)
    predicted_classes_num = np.argmax(predict_x, axis=1)
    predicted_classes = []
    y_test_classes = []

    # convert the numerical labels back to categorical labels
    for i in range(len(predicted_classes_num)):
        predicted_idx = predicted_classes_num[i]
        predicted_classes.append(categ[predicted_idx])

        y_test_idx = y_test[i]
        y_test_classes.append(categ[y_test_idx])

    # create a pandas dataframe (table) listing predicted label vs true label both categorically and numerically
    df = pd.DataFrame({'Predicted': predicted_classes, 'Label': y_test_classes,
                       'Predicted num': predicted_classes_num, 'Label num': y_test})
    print(f"\n{df}\n")

    # display all images in the testing set with true and predicted label in the title
    count = len(x_test)

    # calculate the number of rows needed for the images to be displayed in 10 columns
    nrows = (count + 9) // 10
    fig, axes = plt.subplots(nrows, 10, figsize=(18, 7))

    if nrows == 1:
        axes = [axes]

    for n, img in enumerate(x_test):
        # point at the position of the subplot
        ax = axes[n // 10][n % 10]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Label: {y_test_classes[n]}\n Pred: {predicted_classes[n]}")
        ax.axis('off')

    for n in range(count, nrows * 10):
        ax = axes[n // 10][n % 10]
        ax.axis('off')

    plt.subplots_adjust(left=0.05,
                        right=0.95,
                        wspace=0.2)
    plt.show()

    # return a table of precision, recall, f1-score, and support
    # precision: what proportion of the model prediction for the label was accurate
    # recall: what proportion of the true label was accurately identified
    # f1-score: harmonic mean of precision and recall
    # support: number of elements in each label
    print(classification_report(y_test_classes, predicted_classes, target_names=categ, zero_division='warn'))

def execution(train_path, test_path, model_name):
    # show the categ_num conversion dict
    categ_num_dict_ = categ_num_cov(train_path)
    print(f"\nCategories and the corresponding numerical label: \n{categ_num_dict_}\n")

    # define and show df_train
    df_train = img_df(train_path)
    print(f"Table of files and labels: \n{df_train}\n")

    # visualise dataset from the given path
    show_dataset(train_path)
    X, Y = x_y_dataset(df_train)

    # split the dataset into training and validation
    # random_state can be set to a specific integer for reproducible shuffling output
    # train_size defines the proportion of the elements taken in the train set (first array)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=0, train_size=0.75)

    # testing the datasets
    test_dataset_size(x_train, y_train)
    test_dataset_size(x_val, y_val)

    # training and saving the model
    training, model = train_model(x_train, y_train, x_val, y_val)
    model.save(model_name)

    # graphical display on the training accuracy/loss as it progresses through multiple epochs
    graph_training(training)

    # evaluating the model accuracy with a new testing dataset
    model_eval(test_path, model)