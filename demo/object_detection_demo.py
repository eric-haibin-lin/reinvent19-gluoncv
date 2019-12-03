# Run an object detection model on your webcam
# ==================================================
# This demo will shows how to play with pre-trained object detection models by running
# them directly on your webcam video stream.
#     - This tutorial has only been tested in a MacOS environment
#     - Python packages required: cv2, matplotlib
#     - You need a webcam :)
#     - Python compatible with matplotlib rendering, installed as a framework in MacOS,
#       see guide `here <https://matplotlib.org/faq/osx_framework.html>`__

# Loading the model and webcam
import time
import cv2
import gluoncv as gcv
import mxnet as mx

# In this tutorial we use ``ssd_512_mobilenet1.0_coco``, a snappy network with good accuracy that should be
# well above 1 frame per second on most laptops. Feel free to try a different model from
# the `Gluon Model Zoo <../../model_zoo/detection.html>`__ !
# Load the model
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True)
# Compile the model for faster prediciton
net.hybridize()

# We create the webcam handler in opencv to be able to acquire the frames:
# Load the webcam handler
cap = cv2.VideoCapture(0)
time.sleep(1) ### letting the camera autofocus

# Detection loop
# --------------
# The detection loop consists of four phases:
# * loading the webcam frame
# * pre-processing the image
# * running the image through the network
# * updating the output with the resulting predictions

axes = None
NUM_FRAMES = 2000 # you can change this
for i in range(NUM_FRAMES):
    # Load frame from the camera
    ret, frame = cap.read()

    # Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, ftransform_testrame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

    # Run frame through network
    class_IDs, scores, bounding_boxes = net(rgb_nd)

    # Display the result
    img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
    gcv.utils.viz.cv_plot_image(img)
    cv2.waitKey(1)

# We release the webcam before exiting the script
cap.release()
cv2.destroyAllWindows()
