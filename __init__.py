import cv2
import os
import random
import cv2
import numpy as np
from PIL import Image
import os
import progress
from progress.bar import Bar

FACE_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'facedata' )

class UserFace: 
    
    def __init__(self,
            face_id) : 
        self.none = None 
        self.face_id = face_id
        self.data = {}
        self.data['data_available'] = False
        self.data['facemodel'] = False 
        self.create_initials()

    def create_initials(self) : 
        self.USER_FACE_FOLDER = os.path.join(FACE_DATA , str(self.face_id))
            
        if os.path.isdir(FACE_DATA) == False :
            os.mkdir(FACE_DATA) 
        if os.path.isdir(self.USER_FACE_FOLDER) == False : 
            os.mkdir(self.USER_FACE_FOLDER) 
        else :
            self.data['data_available'] = True
        
            if os.path.isfile(os.path.join(self.USER_FACE_FOLDER , "trainer.yml" )) :
                self.data['facemodel'] = True 

            
    def get_user_face_detected(self , show = True , frames_to_analyze = 100 ) : 
        cam = cv2.VideoCapture(0)
        cam.set(3, 1920) # set video width
        cam.set(4, 1080) # set video height
        cascadePath = os.path.join(os.path.dirname(os.path.realpath(__file__)) , "haarcascade_frontalface_default.xml" )
        face_detector = cv2.CascadeClassifier(cascadePath)

        # For each person, enter one numeric face id
        # Initialize individual sampling face count
        count = 0
        bar = Bar('Getting Your Face Data', max=frames_to_analyze)
        while(True):
            ret, img = cam.read()
            #img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 1 : 
                for (x,y,w,h) in faces:

                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                    count += 1
                    FILENAME = os.path.join(FACE_DATA , str(self.face_id) ,  str(count) + ".jpg")
                    if os.path.isfile(FILENAME) : 
                        FILENAME = os.path.join(FACE_DATA, str(self.face_id) ,  str(random.randint(1,9999)) + ".jpg")
                    # Save the captured image into the datasets folder
                    #cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imwrite( FILENAME , gray[y:y+h,x:x+w])
                    if show == True : 
                        cv2.imshow('image', img)
                    bar.next()
                WAIT = 10
            else : 
                if show == True : 
                    cv2.imshow('image', gray)
                    WAIT = 1
                
            
            k = cv2.waitKey(WAIT) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= frames_to_analyze: # Take 30 face sample and stop video
                break
        # Do a bit of cleanup.
        bar.finish()
        cam.release()
        cv2.destroyAllWindows()

    def analyse_face_data(self) : 

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        cascadePath = os.path.join(os.path.dirname(os.path.realpath(__file__)) , "haarcascade_frontalface_default.xml" )

        detector = cv2.CascadeClassifier(cascadePath)
        

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []
            bar = Bar('Analyzing Your Face', max=len(imagePaths))
            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                #pic_id = int()
                #print ()
                id = int(os.path.split(imagePath)[-1].split(".")[0])
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
                bar.next()
            bar.finish()
            return faceSamples,ids

        faces,ids = getImagesAndLabels(self.USER_FACE_FOLDER)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        #recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
        recognizer.write(os.path.join(self.USER_FACE_FOLDER , "trainer.yml" )) # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
    def detect_face(self , recognition_confidence = 65 , fast_recognize = True  , show = False , max_retries = False) : 
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        recognizer.read(os.path.join(self.USER_FACE_FOLDER , "trainer.yml" ))
        cascadePath = os.path.join(os.path.dirname(os.path.realpath(__file__)) , "haarcascade_frontalface_default.xml" )
        faceCascade = cv2.CascadeClassifier(cascadePath)

        font = cv2.FONT_HERSHEY_SIMPLEX

        #iniciate id counter
        id = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
        #names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W'] 

        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 1920) # set video width
        cam.set(4, 1080) # set video height

        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        attempts = 0 

        while True:

            ret, img =cam.read()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )

            for(x,y,w,h) in faces:
                attempts += 1

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                if (confidence < recognition_confidence):
                    id = self.face_id
                    confidence_pers = "  {0}%".format(round(100 - confidence))
                    if fast_recognize and confidence < recognition_confidence :  
                        cam.release()
                        cv2.destroyAllWindows()
                        return (True , self.face_id , attempts)                       
                else:
                    id = "unknown"
                    confidence_pers = "  {0}%".format(round(100 - confidence))
                
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence_pers), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            if show : 
                cv2.imshow('camera',img) 

                k = cv2.waitKey(10) & 0xff 
                if k == 27:
                    break
            if (max_retries) != False : 
                if attempts > max_retries : 
                    return False
