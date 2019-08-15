# PythonFaceDetection
Python FaceRecongnition (Capture,Analyze,Recognize) Via Webcam/Camera 
# Usage 
### Initial User 
    username = str('jimmy')
    user = UserFace(username)
### Capture New Person Face (Step1)
    username = str('jimmy')
    user = UserFace(username)
    user.get_user_face_detected(frames_to_analyze = 1000)
  * _frames_to_analyze_ : are the numbers of photo captures (not required) 
  * _show_ : show the camera (not required) 
### Analyze User Data (Step2)   
    username = str('jimmy')
    user = UserFace(username)
    user.analyse_face_data()
* NOTE : username should be **captured** before analyzing 

### Detect User (Step3)
    username = str('jimmy')
    user = UserFace(username)
    user.analyse_face_data()
  * NOTE : username should be **captured** and **analyzed** before 
  * _recognition_confidence_ : Confidence (Detection in Lower than 40 could have problems. Recommended 40-60) 
  * _show_ : show the camera (not required) 
  **fast_recognize** : True (return value if detected)
      * Turn this of while testing 
  * _fast_recognize_ : return (True , face_id , number of attempts for detection) if the user is identified if not it would return (False , None , max_retries) (not required) 
  * _max_retries_ : Max retries in case of **fast_recognize** (not required) 

