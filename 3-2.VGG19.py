
# This is second step of the entire script: Bench Mark VGG19

from tensorflow.keras.applications.vgg19 import VGG19

n_epochs = 5
batch_size = 500

# VGG19 Application
model_v = MobileNet(include_top=True, weights='imagenet', input_shape=(im_size, im_size, 1), classes = n_class)

model_v.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_v.summary()

# Fitting VGG19
history_v = model_v.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size,
                        validation_split = .2, verbose = True)

# Train and validation curves with VGG19
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history_v.history['loss'], color = 'b', label = 'Train Loss')
ax1.plot(history_v.history['val_loss'], color = 'm', label = 'Valid Loss')
ax1.legend(loc = 'best')

ax2.plot(history_v.history['acc'], color = 'b', label = 'Train Accuracy')
ax2.plot(history_v.history['val_acc'], color = 'm', label = 'Valid Accuracy')
ax2.legend(loc = 'best')
