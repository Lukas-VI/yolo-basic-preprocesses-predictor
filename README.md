# yolo-basic-preprocesses-predictor

provide a new way to run yolo predict


To improve the inference accuracy of YOLOv8, we can preprocess the images.
The inference process consists of the following steps:
1. Perform inference on the large image to get predicted bounding boxes.
2. Crop the bounding boxes to obtain small image patches.
3. Preprocess the small image patches and perform inference to get predicted class information.
4. Merge the bounding boxes and class information to get the final prediction results.
We need to load two models: a large model (BOX) for object detection and a small model (CLS) for classification.


the file #yolo-basic-preprocesses.py
run your predict task here and save the result in a new file
rembember to change the path of your image and your weight file

the file #yolo-train-preprocesser.py
when you train your sub-model, you can use this script to canny edge detection and cut the image into small pieces,for a special cls modle.

the dictory PTs
contain the preprocess code for 2 tasks,one for detection the location and one for classification.

the 222.jpg is a sample image for test.

✨Please star this repository if it helps you✨
        🚀 and please report issues🚀
        
If you have any questions, please contact me.
