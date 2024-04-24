import tensorflow as tf
import cv2

# location of image
jpg_path="images/sleepy_cat.jpg"
# img_path="images/DLI_Header.png"
resize_factor = 0.25

def test_tf_gpu():
    # print(tf.config.list_physical_devices('GPU'))
    devlist = tf.config.list_physical_devices('GPU')
    assert "GPU" in devlist[0]

def test_cv_jpg_loaded():
    # load image to NumPy array
    img = cv2.imread(jpg_path)
    (h, w) = img.shape[:2]
    print("Height = {}  Width = {}".format(h,w))
    assert h==1200
    assert w==1800

def test_cv_jpg_resize():
    img = cv2.imread(jpg_path)
    (h, w) = img.shape[:2]
    # print("Height = {}  Width = {}".format(h,w))
    if (h is not None) and (w is not None):
        # Resize using x and y factors
        resized_img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, 
                                    interpolation=cv2.INTER_LANCZOS4)
        (h1, w1) = resized_img.shape[:2]
        assert h1==h*resize_factor
        assert w1==w*resize_factor
        # cv2.imshow("Resized Image ", resized_img)
    else:
        assert 0, "Failed to load jpeg image\n"