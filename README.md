# Nhận dạng một số loại tiền Việt Nam

## ***Content***
 - [***Enviroment***](#enviroment)
 - [***Dataset***](#dataset)
 - [***Model***](#model)
 - [***Train***](#train)
 - [***Demo***](#demo)

## ***Enviroment***
 - *Python 3.8*
 - *Matplotlib 3.5*
 - *Numpy 1.21*
 - *OpenCV 4.0*
 - *Tensorflow 2.3*
 - *Keras 2.4*
 - *Sklearn 1.0*

## ***Dataset***
 - Bộ data tự chuẩn bị với 6 class: {Không có tiền, 5000, 10 000, 20 000, 50 000, 100 000}
 - Mỗi class chứa 890 ảnh do nhóm tự chụp từ camera laptop, kích thước (128, 128, 3)
 - Thông qua một số phép augmentation cho ảnh, ta có bộ data như sau:
 <a href="https://drive.google.com/file/d/1q-u1lQBWLfJMDClWmOj3Bb73dXRHp_1s/view?usp=sharing">Download</a>

## ***Model***
 - Xem cấu trúc mô hình:  ```python model.py```
## ***Train***
 - Chạy file data_gen.py để lưu data vào file .pix giúp load data nhanh hơn.
 ``` python data_gen.py```
 - Train model: ```python train.py```
 - Model đã được nhóm train tại đây: <a href="https://drive.google.com/file/d/1DJ_TQqyLcCYFUBTOgwwhWUwKeOLoFDAv/view?usp=sharing"> Download </a>

## ***Demo***
 - Chạy file demo để in ra confusion matrix và chạy demo bằng camera: ``` python demo.py```