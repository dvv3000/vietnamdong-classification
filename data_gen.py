from libs import *
from configs import *

def get_data(label):
    if not os.path.exists(DATA_PATH + "/" + str(label)):
        os.mkdir(DATA_PATH + "/" + str(label))


    cam = cv2.VideoCapture(0)

    # Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
    count = 0
    while(True):
        # Capture frame-by-frame
        count += 1

        ret, frame = cam.read()

        if not ret:
            continue

        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

        # Hiển thị
        cv2.imshow('frame', frame)

        index = (count - 60) / 2
        # Lưu dữ liệu

        if count >= 60 and count % 2 == 0:
            
            print("Số ảnh capture = ", index)
            cv2.imwrite(DATA_PATH + "/" + str(label) + "/" + str(index) + ".png", frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or index == NUMBER_IMAGES:
            break



def save_data(raw_folder=DATA_PATH):

    dest_size = IMAGE_SHAPE[0:2]
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    
    # Lặp qua các folder con trong thư mục raw
    for folder in CLASS_NAME:
        print("Folder=", folder)
        for file in os.listdir(raw_folder + "/" + folder):
            if file != '.DS_Store':
                pixels.append(cv2.resize(cv2.imread(raw_folder + "/" + folder + "/" + file), dsize=dest_size))
                labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels.shape)

    file = open(DATA_FILE, 'wb')
    # dump information to that file
    pickle.dump((pixels,labels), file)
    # close the file
    file.close()




if __name__ == '__main__':
    # get_data(100000)
    save_data()

    