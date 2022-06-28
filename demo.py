from configs import *
from libs import *
from model import *
from train import *


def demoByCam(model):
    cap = cv2.VideoCapture(0)


    while(True):
        # Capture frame-by-frame
        #
        ret, image_org = cap.read()
        if not ret:
            continue
        image_org = cv2.resize(image_org, dsize=None, fx=0.5, fy=0.5)
        # Resize
        image = image_org.copy()
        image = cv2.resize(image, dsize=(128, 128))
        image = image / 255
        # Convert to tensor
        image = np.expand_dims(image, axis=0)

        # Predict
        predict = model.predict(image)
        print("This picture is: ", CLASS_NAME[np.argmax(predict[0])], (predict[0]))
        print(np.max(predict[0],axis=0))
        if (np.max(predict[0])>= 0.8):

            # Show image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1.5
            color = (0, 255, 0)
            thickness = 2

            cv2.putText(image_org, CLASS_NAME[np.argmax(predict[0])], org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Picture", image_org)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def showPredict(model, dataset):
    ncols = 5
    nrows = int(len(dataset) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize = (20, 50))
    for i in range(nrows):
        for j in range(ncols):
            image = dataset[i+j]
            image = image / 255
            # Convert to tensor
            image = np.expand_dims(image, axis=0)
            predict = model.predict(image)

            img = cv2.cvtColor(np.float32(image[0]), cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            axs[i, j].set_title(CLASS_NAME[np.argmax(predict[0])])
    
    plt.show()

if __name__ == "__main__":
    model = get_model()
    model.load_weights(WEIGHT_FILE)

    # calculate the confusion matrix
    X, y = load_data()
    X_train, X_t, y_train, y_t = train_test_split(X, y, test_size=0.2, random_state=30)
    X_test, X_val, y_test, y_val = train_test_split(X_t, y_t, test_size=0.75, random_state=30)
    confusionMatrix(model, X_test / 255, y_test)


    # demoByCam(model)
    showPredict(model, X_test[0:30])


    
   