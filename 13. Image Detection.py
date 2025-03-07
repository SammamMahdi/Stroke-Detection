import cv2
import mediapipe as mp
import pickle
import numpy as np
import os


# Load the pre-trained Voting Classifier model
with open('Voting_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.7)
FACE_INDEXES = {
    # "silhouette": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    #                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    #                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],  # 11
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],  # 10
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],  # 11
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],  # 11
    # 43
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],  # 7
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],  # 9
    "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],  # 7
    "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],  # 9
    "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],  # 7
    "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],  # 9
    "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],  # 9
    # 57
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],  # 8
    "rightEyebrowLower": [35, 124, 46, 53, 52, 65],  # 6
    # 14
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],  # 7
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],  # 9
    "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],  # 7
    "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],  # 9
    "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],  # 7
    "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],  # 9
    "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],  # 9
    # 57
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],  # 8
    "leftEyebrowLower": [265, 353, 276, 283, 282, 295],  # 6
    # 14
    "midwayBetweenEyes": [168],  # 1
    "noseTip": [1],  # 1
    "noseBottom": [2],  # 1
    "noseRightCorner": [98],  # 1
    "noseLeftCorner": [327],  # 1
    # 5
    "rightCheek": [205, 137, 123, 50, 203, 177, 147, 187, 207, 216, 215, 213, 192, 214, 212, 138, 135, 210, 169],  # 19
    "leftCheek": [425, 352, 280, 330, 266, 423, 426, 427, 411, 376, 436, 416, 432, 434, 422, 430, 364, 394, 371]  # 19
    # 38
}
# Define landmarks to exclude based on your drop columns
EXCLUDED_LANDMARKS_INDEXES = [
    130, 25, 110, 24, 23, 22, 26, 112, 243,  # rightEyeLower1
    246, 161, 160, 159, 158, 157, 173,  # rightEyeUpper0
    33, 7, 163, 144, 145, 153, 154, 155, 133,  # rightEyeLower0
    372, 340, 346, 347, 348, 349, 350, 357, 465,  # leftEyeLower3
    226, 31, 228, 229, 230, 231, 232, 233, 244,  # rightEyeLower2
    113, 225, 224, 223, 222, 221, 189,  # rightEyeUpper2
    467, 260, 259, 257, 258, 286, 414,  # leftEyeUpper1
    446, 261, 448, 449, 450, 451, 452, 453, 464,  # leftEyeLower2
    359, 255, 339, 254, 253, 252, 256, 341, 463,  # leftEyeLower1
    342, 445, 444, 443, 442, 441, 413,  # leftEyeUpper2
    247, 30, 29, 27, 28, 56, 190,  # rightEyeUpper1
    35, 124, 46, 53, 52, 65,  # rightEyebrowLower
    265, 353, 276, 283, 282, 295,  # leftEyebrowLower
    466, 388, 387, 386, 385, 384, 398,  # leftEyeUpper0
    1, 2, 168, 98, 327  # nose landmarks and other exclusions
]
print(len(EXCLUDED_LANDMARKS_INDEXES))


# Function to normalize landmarks based on reference point and scale
def normalize_landmarks(landmarks, reference_point, scale_factor):
    centered_landmarks = landmarks - reference_point
    normalized_landmarks = centered_landmarks / scale_factor
    return normalized_landmarks


# Function to calculate scale factor using the distance between two reference points
def get_scale_factor(landmarks, reference_points):
    point_a = landmarks[reference_points[0]]
    point_b = landmarks[reference_points[1]]
    distance = np.linalg.norm(point_a - point_b)
    return distance


# Function to extract and preprocess landmarks from an image
def get_landmarks(image):
    face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
    nose_tip_index = 1  # Index for the nose tip landmark
    right_eye_outer_corner_index = 33  # Index for the right eye outer corner
    left_eye_outer_corner_index = 263  # Index for the left eye outer corner

    data = []

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            all_landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks.landmark])

            # Find the nose tip as the reference point
            nose_tip_point = all_landmarks[nose_tip_index]

            # Calculate scale factor using the distance between two points (e.g., between the eyes)
            scale_factor = get_scale_factor(all_landmarks, [right_eye_outer_corner_index, left_eye_outer_corner_index])

            # Normalize landmarks using the nose tip as the reference point and the calculated scale factor
            normalized_landmarks_np = normalize_landmarks(all_landmarks, nose_tip_point, scale_factor)

            for landmarks_group, indexes in FACE_INDEXES.items():
                for index in indexes:
                    if index not in EXCLUDED_LANDMARKS_INDEXES:
                        x, y = normalized_landmarks_np[index]
                        ratio = y / x if x != 0 else 0
                        data.append(ratio)

    return data


# Function to detect stroke in images within a folder using the pre-trained model
def detect_stroke_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png")):  # Check for both .jpg and .png files
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Extract landmarks
            features = get_landmarks(image)
            if features and len(features) == 115:  # Ensure feature length matches what the model expects
                features = np.array(features).reshape(1, -1)
                # Use the pre-trained model for prediction
                prediction = model.predict(features)
                label = "Stroke" if prediction[0] == 1 else "Non-Stroke"
                print(f"{filename}: {label}")
            else:
                print(f"{filename} {len(features)}did not have the correct number of features. Skipping...")


# Specify the path to your images folder
folder_path = "pictures"  # Replace with your folder path
detect_stroke_in_folder(folder_path)
