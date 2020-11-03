import os
import sys

import face_recognition
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from face_recognition.face_detection_cli import process_images_in_process_pool, image_files_in_folder
from face_recognition.face_recognition_cli import scan_known_people, test_image


# def face_recognition_new(image_path):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "/home/gokul/Documents/image-detection/haarcascade_frontalface_default.xml")
#     KNOWN_FACES_DIR = "/home/gokul/Documents/image-detection/known_faces"
#     # video = cv2.VideoCapture('/home/gokul/Videos/Webcam/raj.webm')
#
#     known_faces = []
#     known_names = []
#
#     dictionary = {}
#
#     for name in os.listdir(KNOWN_FACES_DIR):
#       dictionary[name] = False
#       for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
#         image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
#         encoding = face_recognition.face_encodings(image)[0]
#         known_faces.append(encoding)
#         known_names.append(name)
#
#     req_image = cv2.imread(image_path)
#     locations = face_recognition.face_locations(req_image , model = 'cnn')
#     encodings = face_recognition.face_encodings(req_image , locations)
#     #image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
#
#     for face_encoding,face_location in zip(encodings,locations):
#         results = face_recognition.compare_faces(known_faces , face_encoding , 0.5)
#         match = None
#         if True in results:
#           match = known_names[results.index(True)]
#           print(f"Match found : {match}")
#         else:
#           print(f"Match Not found")

directory = '/home/gokul/Documents/image-detection/known_faces/'
def check_face(image_to_check, cpus=4, tolerance=0.6, show_distance=False):
    for i in os.listdir(directory):
        known_names, known_face_encodings = scan_known_people(image_files_in_folder(directory))

        # Multi-core processing only supported on Python 3.4 or greater
        if (sys.version_info < (3, 4)) and cpus != 1:
            # click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
            cpus = 1

        # if os.path.isdir(image_to_check):
        #     if cpus == 1:
        #         [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files_in_folder(image_to_check)]
        #     else:
        #         process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
        # else:
        result = test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)
        return result

# def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
#     unknown_image = face_recognition.load_image_file(image_to_check)
#
#     # Scale down image if it's giant so things run a little faster
#     if max(unknown_image.shape) > 1600:
#         pil_img = PIL.Image.fromarray(unknown_image)
#         pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
#         unknown_image = np.array(pil_img)
#
#     unknown_encodings = face_recognition.face_encodings(unknown_image)
#
#     for unknown_encoding in unknown_encodings:
#         distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
#         result = list(distances <= tolerance)
#
#         if True in result:
#             [print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
#         else:
#             print_result(image_to_check, "unknown_person", None, show_distance)
#
#     if not unknown_encodings:
#         # print out fact that no faces were found in image
#         print_result(image_to_check, "no_persons_found", None, show_distance)