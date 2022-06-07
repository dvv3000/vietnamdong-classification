from libs import *
from configs import *
from model import *


def load_data():
    file = open(DATA_FILE, 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print("images shape", pixels.shape)
    print("labels shape", labels.shape)


    return pixels, labels

def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    plt.savefig('roc.png')

def lr_scheduler(epoch, lr):
    if epoch > 0.75 * EPOCHS:
        return 0.001
    elif epoch > 0.5 * EPOCHS:
        return 0.01
    return lr

def confusionMatrix(model, X_test, y_test):
    Y_pred = model.predict(X_test)

    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, cmap="rocket_r", fmt=".01f",annot_kws={'size':16}, annot=True, square=True, xticklabels=CLASS_NAME, yticklabels=CLASS_NAME)
    ax.set_ylabel('Actual', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.show()
    
if __name__ == '__main__':
    random.seed(42)
    X, y = load_data()
    # random.shuffle(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Train images shape", X_train.shape)
    print("Train label shape", y_train.shape)
  
    for r in range(10):
        print(CLASS_NAME[np.argmax(y_test[r])])
        cv2.imshow('abc', X_test[r])
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
    # model = getDenseNet()
    model = get_model()
    model.summary()
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    filepath = WEIGHT_PATH + "/" + "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lrScheduler = LearningRateScheduler(lr_scheduler, verbose=1)
    callbacks_list = [checkpoint]


    train_datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range = [0.2, 1.5],
            fill_mode='nearest',
            rescale=1./255,
            )

    test_datagen = ImageDataGenerator(rescale=1./255,)

    history = model.fit(train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                    epochs = EPOCHS,
                                    validation_data = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
                                    callbacks = callbacks_list)

    # plot_model_history(history)
    confusionMatrix(model, X_test/255, y_test)