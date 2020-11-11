import environ

env = environ.Env()
env.read_env('.env')

KNOWN_FACES_DIR = env('KNOWN_FACES_DIRECTORY')
all_face_encodings = {}
import os
import face_recognition
import pickle
if os.path.exists('dataset_faces.dat'):
    os.remove('dataset_faces.dat')

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        print(filename)
        if not face_recognition.face_encodings(image):
            print("\n \n \n Please Insert a Valid Image in {} {} \n \n \n".format(name,filename))
            os.remove(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            break
        all_face_encodings[name] = face_recognition.face_encodings(image)[0]
with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)

