import logging
import os
import pickle

import face_recognition
import numpy as np
from django.shortcuts import render
from mtcnn.mtcnn import MTCNN
# Create your views here.
from rest_framework import generics, status, permissions
from rest_framework.response import Response

# from Faceapp.utils import check_face

logger = logging.getLogger(__name__)

class FaceRecognitionView(generics.GenericAPIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = []
    def post(self,request):
        try:
            data = request.data
            logger.info('Request Payload {}'.format(data))
            patient_photo = data.get('patient_photo')
            if not patient_photo:
                return Response({'status': 'fail', 'message': 'Please Choose a Patient Photo'},
                                status=status.HTTP_400_BAD_REQUEST)

            with open('/home/gokul/dataset_faces.dat', 'rb') as f:
                all_face_encodings = pickle.load(f)

            unknown_image = face_recognition.load_image_file(patient_photo)
            print("unknown_image ",unknown_image)
            if not face_recognition.face_encodings(unknown_image):
                return Response({'status': 'fail', 'message': 'Cant Detect Face first'})
            # unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
            mtcnn = MTCNN()
            detected_face = ''
            faces = mtcnn.detect_faces(unknown_image)
            for face in faces:
                x,y,z,a = face['box']
                detected_face = unknown_image[y:y+a,x:x+z]
                print("detected_face first ",detected_face)
            print("detected_face second ",detected_face)
            if detected_face == '':
                return Response({'status': 'fail', 'message': 'Cant Detect Face'})
            locations = face_recognition.face_locations(detected_face, model='cnn')
            if not locations:
                return Response({'status': 'fail', 'message': 'Cant Detect Face second'})
            encodings = face_recognition.face_encodings(detected_face, locations)
            print("encodings",encodings)
            face_names = list(all_face_encodings.keys())
            face_encodings = np.array(list(all_face_encodings.values()))
            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(face_encodings, face_encoding, 0.45)
                match = None
                if True in results:
                    # print("all face encodings - ",face_encodings)
                    # print("face encoding - ",face_encoding)
                    match = face_names[results.index(True)]
                    print(f"Match found : {match}")
                    return Response({'status': 'success', 'message': 'Face Recognised Successfully', 'data': match})
                else:
                    print("Match Not Found")
                    return Response({'status': 'success', 'message': 'Face Recognised Successfully', 'data': "Match Not Found"})

        except Exception as e:
            logger.exception('Exception {}'.format(e.args))
            return Response({'status': 'fail', 'message': 'Something went wrong. Please try again later'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

