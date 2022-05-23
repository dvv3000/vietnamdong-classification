from libs import *
from configs import *
from model import get_model


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
    #plt.show()
    plt.savefig('roc.png')

def lr_scheduler(epoch, lr):
    if epoch > 0.75 * EPOCHS:
        return 0.001
    elif epoch > 0.5 * EPOCHS:
        return 0.01
    return lr


    
if __name__ == '__main__':
    random.seed(42)
    X, y = load_data()
    # random.shuffle(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Train images shape", X_train.shape)
    print("Train label shape", y_train.shape)


    model = get_model()
    model.summary()

    filepath = WEIGHT_PATH + "/" + "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lrScheduler = LearningRateScheduler(lr_scheduler, verbose=1)
    callbacks_list = [checkpoint, earlystopping]


    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
        rescale = 1./255,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        brightness_range = [0.2, 1.5],
        fill_mode = "nearest")

    aug_val = ImageDataGenerator(rescale = 1./255)

    history = model.fit(aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                epochs = EPOCHS,
                                validation_data = aug_val.flow(X_test, y_test, batch_size=BATCH_SIZE),
                                callbacks = callbacks_list)

    plot_model_history(history)

    model.save("model.h5")