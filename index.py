import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Deep Learning Libraries
from keras.api.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

picture_size = 48
folder_path = "D:/Python Programs/Programs/AI Image Visualization/archive/images"
expression = "disgust"
batch_size = 64
no_of_classes = 7
epochs = 20


plt.figure(figsize=(12, 12))
for i in range(1, 10):
    plt.subplot(3, 3, i)
    img_path = os.path.join(folder_path, "train", expression, os.listdir(os.path.join(folder_path, "train", expression))[i])
    img = load_img(img_path, target_size=(picture_size, picture_size))
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
plt.show()


train_data_generator = ImageDataGenerator()
test_data_generator = ImageDataGenerator()

train_set = train_data_generator.flow_from_directory(
    os.path.join(folder_path, "train"),
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_set = test_data_generator.flow_from_directory(
    os.path.join(folder_path, "validation"),
    target_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

emotion_model = Sequential([
    # 1st Layer
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),

    # 2nd Layer
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 3rd Layer
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # 4th Layer
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),

    # Fully Connected Layer
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(no_of_classes, activation='softmax')
])


emotion_model.summary()


checkpoint = ModelCheckpoint("./model.keras", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

initial_learning_rate = 0.0005
optimizer = Adam(learning_rate=initial_learning_rate)

emotion_model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['recall','accuracy','precision','f1_score'])



num_of_train_img = train_set.n
num_of_test_img = test_set.n

history = emotion_model.fit(
    train_set,
    steps_per_epoch=num_of_train_img // batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=num_of_test_img // test_set.batch_size,
    callbacks=callbacks_list
)



plt.figure(figsize=(20, 10))

# Plot Loss
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer: Adam', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')

plt.show()