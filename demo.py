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
        if (np.max(predict[0])>= 0.6):

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



if __name__ == "__main__":
    # model = getDenseNet()
	model = get_model()
	model.load_weights(WEIGHT_FILE)


	X, y = load_data()
	X_train, X_t, y_train, y_t = train_test_split(X, y, test_size=0.2, random_state=42)
	X_test, X_val, y_test, y_val = train_test_split(X_t, y_t, test_size=0.5, random_state=42)
	confusionMatrix(model, X_train / 255, y_train)

    # demoByCam(model)


    
   