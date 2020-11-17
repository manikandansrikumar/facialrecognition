import os
import random
import shutil

import face_recognition
import cv2
import numpy as np
import pickle


# Get a reference to webcam #0 (the default one)
from rest_framework import generics, status
from rest_framework.response import Response
from FaceRecognition.settings import KNOWN_FACE_DIRECTORY


class RealTimeFaceRecognition(generics.GenericAPIView):
    def post(self,request):
        try:
            # Create if there is no cropped face directory
            dirFace = 'media/cropped_face'
            if not os.path.exists(dirFace):
                os.mkdir(dirFace)
                print("Directory ", dirFace, " Created ")
            else:
                print("Directory ", dirFace, " has found.")

            video_capture = cv2.VideoCapture(0)


            with open('/home/gokul/Documents/FaceRecognition/dataset_faces.dat', 'rb') as f:
                all_face_encodings = pickle.load(f)

            known_face_names = list(all_face_encodings.keys())
            known_face_encodings = np.array(list(all_face_encodings.values()))
            # Initialize some variables
            face_locations = []
            face_encodings = []
            face_names = []
            process_this_frame = True
            dictionary = {}
            while True:
                # Grab a single frame of video
                ret, frame = video_capture.read()

                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.45)
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        print(best_match_index)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            # Save Photo to Cropped Face Folder
                            FaceFileName = "media/cropped_face/" + str(name) + ".jpg"  # folder path and random name image
                            cv2.imwrite(FaceFileName, frame)
                            print('Face saved')
                                # dictionary[match] = True

                        face_names.append(name)

                process_this_frame = not process_this_frame


                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Display the resulting image
                cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()

            # save photo to certain folder
            for filename in os.listdir(dirFace):
                image = cv2.imread(f"{dirFace}/{filename}")
                match = filename.split('.')[0]
                path = os.path.join(KNOWN_FACE_DIRECTORY, match)
                randomnum = random.randint(0, 100)
                cv2.imwrite(f"{path}/{match}{randomnum}.jpg",image)
            # Remove all photos inside cropped face
            shutil.rmtree('media/cropped_face/')
            return Response({'status': 'success', 'message': 'Face Detected Successfully'})

        except Exception as e:
            # logger.exception('Exception {}'.format(e.args))
            return Response({'status': 'fail', 'message': 'Something went wrong. Please try again later'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
