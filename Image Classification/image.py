import tensorflow as tf
import os
import numpy as np
import cv2
import imghdr
import matplotlib
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
DataDir = 'image data'
ImageEx = ['jpeg', 'jpg', 'bmp', 'png']

print(os.listdir(DataDir))
print(os.listdir(os.path.join(DataDir, 'Beyonce',)))

for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir,image_class)):
        print(image)

total_images = 0  # initialize counter variable
for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir, image_class)):
        total_images += 1  # increment counter for each file
print("Total number of images:", total_images)


total_images = 0  # initialize counter variable
removed_images = 0  # initialize counter variable for removed images
for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir, image_class)):
        ImagePath = os.path.join(DataDir, image_class,image)
        try:
            DImage = cv2.imread(ImagePath)
            ImaS = imghdr.what(ImagePath)
            if ImaS not in ImageEx:
                print('Image not in extension list {}'.format(ImagePath))
                os.remove(ImagePath)
                removed_images += 1  # increment counter for each removed file
        except Exception as e:
            print('Issue with image {}'.format(ImagePath))
            # os.remove(ImagePath)  # Uncomment this if you want to remove the problematic images
        total_images += 1  # increment counter for each file

print("Total number of images found:", total_images)
print("Total number of images removed:", removed_images)

# preprocessing the images

ImageData = tf.keras.utils.image_dataset_from_directory('image data', )
print(ImageData)

# data image iterator is ImageItr

# data image iterator is ImageItr
ImageItr = ImageData.as_numpy_iterator()
try:
    Imagebatch = ImageItr.next()
    print(Imagebatch[1].shape)

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))

    for idx, DImage in enumerate(Imagebatch[0][:4]):
        ax[idx].imshow(DImage.astype(int))
        ax[idx].title.set_text(Imagebatch[1][idx])

    plt.show()

    # preprocessing the images
    for image, label in ImageItr:
        print("Processing image:", label)
        try:
            # preprocess the image
            pass
        except Exception as e:
            print('Problematic image:', ImagePath)

except Exception as e:
    print('An error occurred during image preprocessing:', e)



#preprocessing the data

ImageData = tf.concat(list(ImageData), axis=0)
ImageData = ImageData / 255.0


Idata = Idata.map(lambda x, y: (x / 255, y))

# Check the range of the data after scaling
print(Idata.as_numpy_iterator().next()[0].max())
print(Idata.as_numpy_iterator().next()[0].min())

# Display a few images from the dataset
scaled_iterator = Idata.as_numpy_iterator()
Thebatch = scaled_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, DImage in enumerate(Thebatch[0][:4]):
    ax[idx].imshow(DImage.astype(int))
    label = np.argmax(Thebatch[1][idx]) # get the index of the highest value in the one-hot encoded array
    ax[idx].title.set_text(label)
plt.show()

# set the sizes for the train, validation, and test sets
TotalSize = 353
TrainSize = 247
ValSize = 71
TestSize = 35

# load the image dataset
Idata = tf.keras.preprocessing.image_dataset_from_directory(
    'Image data',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.3,
    subset='training',
    seed=123
)

print('Training Size=', TrainSize)
print('Validation Size=', ValSize)
print('Test Size=', TestSize)
print('Total Size=', TrainSize + ValSize + TestSize)

# the skip and take tensorflow function is used so the data from the training won't appear in the validation or test set
Train = Idata.take(TrainSize)
Val = Idata.skip(TrainSize).take(ValSize)
Test = Idata.skip(TrainSize+ValSize).take(TestSize)

# building the model
IModel = Sequential()
IModel.add(Conv2D(8, (3, 3), 2, activation='relu', input_shape=(256, 256, 3)))
IModel.add(MaxPooling2D())
IModel.add(Conv2D(16, (3, 3), 2, activation='relu'))
IModel.add(MaxPooling2D())
IModel.add(Conv2D(8, (3, 3), 2, activation='relu'))
IModel.add(MaxPooling2D())
IModel.add(Flatten())
IModel.add(Dense(256, activation='relu'))
IModel.add(Dense(4, activation='softmax'))

# i used less filter for it to be faster at the expense of accuracy and the stride of 2 instead of 1 for speed over accuracy
# If you have 4 classes of images to classify, you should use Dense(4, activation='softmax') instead of Dense(1, activation='sigmoid') in your final layer.
# The softmax activation function is typically used for multi-class classification problems and produces a probability distribution over the different classes.
# The output of the softmax layer will be a vector of length 4, with each element representing the probability of the corresponding class.





#compile model
IModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#print model summary
print(IModel.summary)

LogDir = 'ImageLogs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LogDir)

hist = IModel.fit(Train, epochs=10, validation_data=Val, callbacks=[tensorboard_callback])
