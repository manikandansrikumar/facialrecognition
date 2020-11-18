import logging
import os
import pickle

import face_recognition
import numpy as np
from PIL import Image
import cv2
from django.conf import settings
from django.shortcuts import render
from mtcnn.mtcnn import MTCNN
# Create your views here.
from rest_framework import generics, status, permissions
from rest_framework.response import Response

# from Faceapp.utils import check_face
from FaceRecognition.settings import KNOWN_FACE_DIRECTORY

logger = logging.getLogger(__name__)

class FaceRecognitionView(generics.GenericAPIView):
    def post(self,request):
        try:
            data = request.data
            logger.info('Request Payload {}'.format(data))
            patient_photo = data.get('patient_photo')
            if not patient_photo:
                return Response({'status': 'fail', 'message': 'Please Choose a Patient Photo'},
                                status=status.HTTP_400_BAD_REQUEST)
            known_face_directory = KNOWN_FACE_DIRECTORY
            with open(settings.DATA_SET_PATH, 'rb') as f:
                all_face_encodings = pickle.load(f)

            unknown_image = face_recognition.load_image_file(patient_photo)

            if not face_recognition.face_encodings(unknown_image):
                return Response({'status': 'fail', 'message': 'Cant Detect Face'},status=status.HTTP_400_BAD_REQUEST)
            # unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
            mtcnn = MTCNN()
            detected_face = ''
            faces = mtcnn.detect_faces(unknown_image)
            try:
                for face in faces:
                    x,y,z,a = face['box']
                    detected_face = unknown_image[y:y+a,x:x+z]
            except:
                return Response({'status': 'fail', 'message': 'Cant Detect Face'},status=status.HTTP_400_BAD_REQUEST)
            # import numpy as np
            if np.array(detected_face).size == 0 or detected_face =='':
                return Response({'status': 'fail', 'message': 'Cant Detect Face'},status=status.HTTP_400_BAD_REQUEST)
            locations = face_recognition.face_locations(detected_face, model='cnn')

            if not locations:
                return Response({'status': 'fail', 'message': 'Cant Detect Face second'},status=status.HTTP_400_BAD_REQUEST)
            encodings = face_recognition.face_encodings(detected_face, locations)

            face_names = list(all_face_encodings.keys())
            face_encodings = np.array(list(all_face_encodings.values()))
            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(face_encodings, face_encoding, 0.45)
                match = None
                if True in results:
                    match = face_names[results.index(True)]
                    print(f"Match found : {match}")
                    # add images to existing folder
                    pil_img = Image.open(patient_photo)
                    np_img = np.array(pil_img)
                    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    path = os.path.join(known_face_directory,match)
                    cv2.imwrite(os.path.join(path, patient_photo.name), img)

                    return Response({'status': 'success', 'message': 'Face Recognised Successfully', 'data': match})
                else:
                    print("Match Not Found")
                    return Response({'status': 'fail', 'message': 'Match Not Found'},status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.exception('Exception {}'.format(e.args))
            return Response({'status': 'fail', 'message': 'Something went wrong. Please try again later'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SavePhotoView(generics.GenericAPIView):
    def post(self,request):
        try:
            data = request.data
            logger.info('Request Payload {}'.format(data))
            patient_photo = data.get('patient_photo')
            patient_photo1 = data.get('patient_photo1')
            patient_photo2 = data.get('patient_photo2')
            patient_photo3 = data.get('patient_photo3')
            patient_photo4 = data.get('patient_photo4')
            patient_id = data.get('patient_id')
            if not patient_id:
                return Response({'status': 'fail', 'message': 'Please Choose a Patient'},
                                status=status.HTTP_400_BAD_REQUEST)

            if not patient_photo or not patient_photo1 or not patient_photo2 or not patient_photo3 or not patient_photo4:
                return Response({'status': 'fail', 'message': 'Please Choose a Patient Photo'},
                                status=status.HTTP_400_BAD_REQUEST)
            create_directory = patient_id
            directory = settings.KNOWN_FACE_DIRECTORY
            path = os.path.join(directory, create_directory)
            try:
                os.mkdir(path)
            except OSError as error:
                pass
                # logger.exception('os error {}'.format(error.args))
                # return Response({'status': 'fail', 'message': "File not Found"})
            for photo in data.values():
                if patient_id == photo:
                    pass
                else:
                    pil_img = Image.open(photo)
                    np_img = np.array(pil_img)
                    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(path, photo.name), img)

            return Response({'status': 'success', 'message': 'Photo Stored Successfully'})

        except Exception as e:
            logger.exception('Exception {}'.format(e.args))
            return Response({'status': 'fail', 'message': 'Something went wrong. Please try again later'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
