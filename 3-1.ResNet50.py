
# This is second step of the entire script: Bench Mark model_ResNet50

from keras.applications.resnet50 import ResNet50

n_epochs = 5
batch_size = 500

# ResNet50 Application
model_r = ResNet50(include_top = True, weights= 'imagenet', input_shape=(im_size, im_size, 3), classes = n_class)

model_r.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_r.summary()

# Fitting ResNet50
history_r = model_r.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size,
                      validation_split = .2, verbose = True)

# Train and validation curves with ResNet50
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history_r.history['loss'], color = 'b', label = 'Train Loss')
ax1.plot(history_r.history['val_loss'], color = 'm', label = 'Valid Loss')
ax1.legend(loc = 'best')

ax2.plot(history_r.history['acc'], color = 'b', label = 'Train Accuracy')
ax2.plot(history_r.history['val_acc'], color = 'm', label = 'Valid Accuracy')
ax2.legend(loc = 'best')
