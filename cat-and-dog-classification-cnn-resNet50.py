import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# import torch

# use gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

train_path = '/kaggle/input/cat-and-dog/training_set/training_set'
test_path = '../input/cat-and-dog/test_set/test_set'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10, shuffle=False)
imgs, labels = next(train_batches)

fig, axes = plt.subplots(1, 10, figsize=(20,20))
axes = axes.flatten()
for img, ax in zip(imgs, axes):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()

print(labels)

# model
model = Sequential()

model.add(ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', 
                metrics=["accuracy"],
                optimizer=Adam(learning_rate=0.0001))

epochs = 5

history = model.fit(x = train_batches,
                    epochs = epochs,
                    verbose = 1)

# loss
plt.plot(history.history['loss'], color='black')
plt.show()

# accuracy
plt.plot(history.history['accuracy'], color='black')
plt.show()

# test
test_imgs, test_labels = next(test_batches)
predictions = model.predict(x=test_batches, verbose=0)

# confusion matrix
confusion_mtx = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
print(confusion_mtx)

classes=['cats', 'dogs']

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(3, 2.5))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.gray_r)
plt.title('Confusion matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = confusion_mtx.max() / 2.
for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    plt.text(j, i, confusion_mtx[i, j],
            horizontalalignment="center",
            color="white" if confusion_mtx[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

# accuracy
accuracy = accuracy_score(test_batches.classes, np.argmax(predictions, axis=-1))
print(f'Accuracy: {accuracy}')