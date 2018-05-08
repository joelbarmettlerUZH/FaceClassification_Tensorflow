from BingImages import BingImages
import cv2
import os
import requests
import shutil
from classifier import classify
import threading
import tarfile
import zipfile
from copy import deepcopy

class FaceClassifier:

    # Constructor: What classes are to classify?
    def __init__(self, *persons):
        self.__persons = persons

    # @PARAM count: number. The number of images to download for each class
    # @PARAM download: boolean. Download new training data to classify the model
    # @PARAM delete:  boolean. Delete existing training data and start from scratch
    def trainModel(self, count=300, download=True, delete=False):

        # Delete existing resources
        if(delete and download):
            try:
                shutil.rmtree("./tf")
                shutil.rmtree("./training_images")
                shutil.rmtree("./tmp")
            except FileNotFoundError:
                pass
        print("--Start training Model")

        #Download cascade face detection resources
        if not os.path.exists("./cascade/haarcascade_frontalface_default.xml"):
            self.__downloadCascade()

        # Download resources for class training and then cut out their faces
        if download:
            self.__resourceDownloader(count)
            self.__cutFaces()

        # Create all needed folders for the classifier
        folders = ["./tf", "./tf/training_data", "./tf/training_output", "./tf/training_data/summaries/basic"]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Retrain the tensorflow model
        print("--Training model")
        os.system("retrain.py --tf/training_data/bottleneck_dir=bottlenecks --model_dir=tf/training_data/inception --summaries_dir=tf/training_data/summaries/basic --output_graph=tf/training_output/retrained_graph.pb --output_labels=tf/training_output/retrained_labels.txt --image_dir=training_images --how_many_training_steps=4000")

    # Cut faces of 
    def __cutFaces(self):
        print("--Cutting out faces")
        i = 0

        # Loop through folders, cut out face and save face into training_images directory
        for person in self.__persons:
            folder = "./tmp/downloaded_images/" + person
            facefolder = "./training_images/" + person
            if not os.path.exists(facefolder):
                os.makedirs(facefolder)
            for file in os.listdir(folder):
                image = cv2.imread(folder + "/" + file)
                # Detecting Faces
                faces = self.__faceDetector(image)
                for face in faces:
                    # Saving the image of the face to the disk
                    cv2.imwrite(facefolder + "/face_{}.jpg".format(i), face)
                    i += 1

    # Downloading haarcascade feature set from github
    def __downloadCascade(self):
        print("Downloading haarcascade for face detection")
        url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
        folder = "./cascade/"
        local_filename = folder + url.split('/')[-1]
        # Check if already exists on users disk
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Stream download dataset to lcoal disk
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    # Detects face in cv2 image via haarcascade
    def __faceDetector(self, img):
        face_cascade = cv2.CascadeClassifier("./face_cascade/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faceCuts = []
        for (x, y, w, h) in faces:
            # Faces are recognized with x-y (top-left point) and width-height
            faceCuts.append(img[y:y + h, x:x + w])
        # Returny images (numpy array) of detected faces
        return faceCuts

    # Downloads resource images with the specified terms
    def __resourceDownloader(self, count=300):
        for person in self.__persons:
            print("--Downloading Resources for '{}'".format(person))
            folder = "./tmp/downloaded_images/" + person
            # Fetch links from Bing Images and download images
            BingImages.download(BingImages(person + " face", count=count, person="portrait").get(), folder)
            counter = 0
            # Rename the files accordingly
            for filename in os.listdir(folder):
                filepath = folder + "/" + filename
                _, file_extension = os.path.splitext(filepath)
                # Remove files that are of wrong type or too small
                if file_extension.lower() not in [".jpg", ".jpeg"] or os.path.getsize(filepath) < 1024*128: # File not jpeg smaller than 128kb
                    os.remove(filepath)
                    continue
                tries = 0
                # Rename all other files with fitting schema "img_X.jpg"
                while(tries < 1000000):
                    try:
                        os.rename(filepath, folder + "/" + "image_" + str(counter) + ".jpg")
                        break
                    # Catch error that a file may exist already with a certain name
                    except FileExistsError:
                        tries += 1
                        counter += 1
                        pass
                counter += 1

    # Train a faster mobile neural network with the intelligence of the inception v3 model
    def fastenNetwork(self, newDataset=True):
        print("Fastening Network")
        lfw_folder = "./lfw_dataset/images"
        # Download datasets
        bigdata_folder = "training_images_bigdata"
        if (not os.path.exists(lfw_folder)) or newDataset:
            self.__downloadLFW()
        # Creates big data folder for saving faces if not existent
        if os.path.exists(bigdata_folder) and not newDataset:
            self.__trainFast()
            return
        elif not os.path.exists(bigdata_folder):
            os.makedirs(bigdata_folder)
        i = 0
        models = {}
        for person in self.__persons:
            models[person.lower()] = 0
        # Train a new dataset if requested, use the existing images otherwise
        if newDataset:
            print("Classifying new images from LFW Dataset")
            for folder in self.__persons:
                if not os.path.exists(bigdata_folder + "/" + folder):
                    os.makedirs(bigdata_folder + "/" + folder)
            # Loop over all the files
            for file in os.listdir(lfw_folder):
                print("Processing {}".format(lfw_folder + "/" + file))
                image = cv2.imread(lfw_folder + "/" + file)
                # Detect faces
                faces = self.__faceDetector(image)
                for face in faces:
                    # Save face temporarily for the classifier
                    cv2.imwrite(bigdata_folder + "/tmpface.jpg", face)
                    # Classify face
                    predictions = classify(bigdata_folder + "/tmpface.jpg", "./tf/training_output/retrained_graph.pb", "./tf/training_output/retrained_labels.txt")
                    # Save image to the classified class if certainty is above 60%, skip image otherwise
                    if predictions[0][1] > .6:
                        cv2.imwrite(bigdata_folder + "/" + predictions[0][0] + "/img_{}.jpg".format(i), face)
                        i += 1
                        models[predictions[0][0]] += 1
                        print("Current prediction status: ", models)

    def trainFast(self):
        # Create all needed folders for the classifier
        folders = ["./tf", "./tf/training_data", "./tf/training_output", "./tf/training_data/summaries/basic"]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
        # Train Model
        print("--Training model")
        os.system("retrain.py --tf/training_data/bottleneck_dir=bottlenecks --model_dir=tf/training_data/inception --summaries_dir=tf/training_data/summaries/basic --output_graph=tf/training_output/retrained_graph.pb --output_labels=tf/training_output/retrained_labels.txt --image_dir=training_images_bigdata --how_many_training_steps=800 --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1 --validation_batch_size=-1 --learning_rate=0.0001 --random_brightness=30")


    def __downloadLFW(self):
        print("Downloading LFW Face Dataset for face detection")
        # Links to all dataset archives
        urls = [
            "http://pics.psych.stir.ac.uk/zips/Aberdeen.zip",
            "http://pics.psych.stir.ac.uk/zips/Iranian.zip",
            "http://pics.psych.stir.ac.uk/zips/pain.zip",
            "http://pics.psych.stir.ac.uk/zips/utrecht.zip",
            "https://www.openu.ac.il/home/hassner/data/lfwa/lfwa.tar.gz"
        ]
        folder = "./lfw_dataset/tmp/"
        # Download all datasets
        for url in urls:
            print("Start downloading {}".format(url))
            local_filename = folder + url.split('/')[-1]
            file_extention = local_filename[local_filename.rfind(".")+1:]
            # Skip if archive already exists
            if not os.path.exists(local_filename):
                print("File extention: {}".format(file_extention))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                # Open up download strem to file
                r = requests.get(url, stream=True)
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print("Download complete. Entzipping now")
                # Unpack zips
                if file_extention == "zip":
                    print("Unzipping file {}".format(local_filename))
                    zippedFile = zipfile.ZipFile(local_filename, 'r')
                    zippedFile.extractall(folder)
                    zippedFile.close()
                # Unpack tars
                else:
                    print("Untarring file {}".format(local_filename))
                    tarredFile = tarfile.open(local_filename, "r:gz")
                    tarredFile.extractall(folder)
                    tarredFile.close()
            else:
                print("Dataset already exists. Skipping it.")

        print("Finished all downloads. Reordering data")
        i = 0
        data_folder = "./lfw_dataset/images/"
        # Refresh folder if already existent
        try:
            shutil.rmtree(data_folder)
        except FileNotFoundError:
            pass
        os.makedirs(data_folder)
        # Copy all files to new location with correct naming schema
        for root, dirs, files in os.walk("./lfw_dataset/tmp/"):
            for file in files:
                filename = os.path.join(root, file)
                _, file_extension = os.path.splitext(filename)
                if file_extension.lower() == ".jpg" and not os.path.exists(data_folder + "img_{}.jpg".format(i)):
                    print("Copy file: {} to img_{}.jpg".format(filename, i))
                    shutil.copyfile(filename, data_folder + "img_{}.jpg".format(i))
                    i += 1
        print("Done setting up the dataset")

    def liveDetect(self):
        filename = "./tmp/face.jpg"

        # Inner function for thread to parallel process image classification according to trained model
        def classifyFace():
            print("Classifying Face")
            prediction = classify(filename, "./tf/training_output/retrained_graph.pb", "./tf/training_output/retrained_labels.txt", shape=224)
            nonlocal text
            text = prediction[0][0]
            print("Finished classifying with text: " + text)

        # Initialize the cascade classifier for detecting faces
        face_cascade = cv2.CascadeClassifier("./face_cascade/haarcascade_frontalface_default.xml")

        # Initialize the camera (use bigger indices if you use multiple cameras)
        cap = cv2.VideoCapture(0)
        # Set the video resolution to half of the possible max resolution for better performance
        cap.set(3, 1920 / 2)
        cap.set(4, 1080 / 2)

        # Standard text that is displayed above recognized face
        text = "unknown face"
        exceptional_frames = 100
        startpoint = (0, 0)
        endpoint = (0, 0)
        color = (0, 0, 255) # Red
        while True:
            # Read frame from camera stream and convert it to greyscale
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces using cascade face detection
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through detected faces and set new face rectangle positions
            for (x, y, w, h) in faces:
                color = (0, 0, 255)
                if not text == "unknown face":
                    color = (0, 255, 0)
                oldstartpoint = deepcopy(startpoint)
                startpoint = (x, y)
                endpoint = (x + w, y + h)
                face = (img[y:y + h, x:x + w])
                # Only reclassify if face was lost for at least half a second (15 Frames at 30 FPS)
                if exceptional_frames > 15 or all(abs(i - j) > 15 for i, j in zip(startpoint, oldstartpoint)):
                    # Save detected face and start thread to classify it using tensorflow model
                    print("Redetect face due to heavy movement")
                    cv2.imwrite(filename, face)
                    threading._start_new_thread(classifyFace, ())
                exceptional_frames = 0

            # Face lost for too long, reset properties
            if exceptional_frames == 15:
                print("Exceeded exceptional frames limit")
                text = "unknown face"
                startpoint = (0, 0)
                endpoint = (1, 1)

            # Draw face rectangle and text on image frame
            cv2.rectangle(img, startpoint, endpoint, color, 2)
            textpos = (startpoint[0], startpoint[1] - 7)
            cv2.putText(img, text, textpos, 1, 1.5, color, 2)
            # Show image in cv2 window
            cv2.imshow("image", img)
            # Break if input key equals "ESC"
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            exceptional_frames += 1

    def liveFaceDetection(self):

        # Initialize the cascade classifier for detecting faces
        face_cascade = cv2.CascadeClassifier("./face_cascade/haarcascade_frontalface_default.xml")

        # Initialize the camera (use bigger indices if you use multiple cameras)
        cap = cv2.VideoCapture(0)
        # Set the video resolution to half of the possible max resolution for better performance
        cap.set(3, 1920 / 2)
        cap.set(4, 1080 / 2)

        # Standard text that is displayed above recognized face
        exceptional_frames = 100
        while True:
            print(exceptional_frames)
            # Read frame from camera stream and convert it to greyscale
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces using cascade face detection
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through detected faces and set new face rectangle positions
            for (x, y, w, h) in faces:
                color = (0, 255, 0)
                startpoint = (x, y)
                endpoint = (x + w, y + h)
                exceptional_frames = 0
                # Draw face rectangle on image frame
                cv2.rectangle(img, startpoint, endpoint, color, 2)

            # Show image in cv2 window
            cv2.imshow("image", img)
            # Break if input key equals "ESC"
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            exceptional_frames += 1

if __name__ == "__main__":
    classifier = FaceClassifier("Male", "Female")
    # classifier.trainModel(100, download=True, delete=True)
    # classifier.liveDetect()
    # classifier.fastenNetwork(newDataset=False)
    # classifier.trainFast()
    # classifier.liveDetect()
    classifier.liveFaceDetection()