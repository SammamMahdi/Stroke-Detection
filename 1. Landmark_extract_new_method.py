import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
from collections import defaultdict

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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


def normalize_landmarks(landmarks, reference_point, scale_factor):
    centered_landmarks = landmarks - reference_point
    normalized_landmarks = centered_landmarks / scale_factor
    return normalized_landmarks


def get_scale_factor(landmarks, reference_points):
    point_a = landmarks[reference_points[0]]
    point_b = landmarks[reference_points[1]]
    distance = np.linalg.norm(point_a - point_b)
    return distance


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

            # Store normalized landmarks for selected features
            for landmarks_group, indexes in FACE_INDEXES.items():
                for index in indexes:
                    normalized_landmark = normalized_landmarks_np[index]
                    data.append((landmarks_group, index, *normalized_landmark))
    return data


folder_path = "Final Data Set Normal"
# folder_path = "FInal Data Set Stroke"
all_landmarks = defaultdict(list)

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".png")):  # Check for both .jpg and .png files
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        landmarks = get_landmarks(image)
        for landmark in landmarks:
            group, index, x, y = landmark
            all_landmarks[filename].extend([y / x])
to_csv = []
for filename, landmarks in all_landmarks.items():
    out = [filename, *landmarks]
    if len(out) == 229:
        to_csv.append(out)
    else:
        print(filename, len(out))
# Convert the list of tuples into a DataFrame
df = pd.DataFrame(to_csv,
                  columns=['Filename', 'lipsUpperOuter_61', 'lipsUpperOuter_185', 'lipsUpperOuter_40',
                           'lipsUpperOuter_39',
                           'lipsUpperOuter_37', 'lipsUpperOuter_0', 'lipsUpperOuter_267', 'lipsUpperOuter_269',
                           'lipsUpperOuter_270', 'lipsUpperOuter_409', 'lipsUpperOuter_291', 'lipsLowerOuter_146',
                           'lipsLowerOuter_91', 'lipsLowerOuter_181', 'lipsLowerOuter_84', 'lipsLowerOuter_17',
                           'lipsLowerOuter_314', 'lipsLowerOuter_405', 'lipsLowerOuter_321', 'lipsLowerOuter_375',
                           'lipsLowerOuter_291', 'lipsUpperInner_78', 'lipsUpperInner_191', 'lipsUpperInner_80',
                           'lipsUpperInner_81', 'lipsUpperInner_82', 'lipsUpperInner_13', 'lipsUpperInner_312',
                           'lipsUpperInner_311', 'lipsUpperInner_310', 'lipsUpperInner_415', 'lipsUpperInner_308',
                           'lipsLowerInner_78', 'lipsLowerInner_95', 'lipsLowerInner_88', 'lipsLowerInner_178',
                           'lipsLowerInner_87', 'lipsLowerInner_14', 'lipsLowerInner_317', 'lipsLowerInner_402',
                           'lipsLowerInner_318', 'lipsLowerInner_324', 'lipsLowerInner_308', 'rightEyeUpper0_246',
                           'rightEyeUpper0_161', 'rightEyeUpper0_160', 'rightEyeUpper0_159', 'rightEyeUpper0_158',
                           'rightEyeUpper0_157', 'rightEyeUpper0_173', 'rightEyeLower0_33', 'rightEyeLower0_7',
                           'rightEyeLower0_163', 'rightEyeLower0_144', 'rightEyeLower0_145', 'rightEyeLower0_153',
                           'rightEyeLower0_154', 'rightEyeLower0_155', 'rightEyeLower0_133', 'rightEyeUpper1_247',
                           'rightEyeUpper1_30', 'rightEyeUpper1_29', 'rightEyeUpper1_27', 'rightEyeUpper1_28',
                           'rightEyeUpper1_56', 'rightEyeUpper1_190', 'rightEyeLower1_130', 'rightEyeLower1_25',
                           'rightEyeLower1_110', 'rightEyeLower1_24', 'rightEyeLower1_23', 'rightEyeLower1_22',
                           'rightEyeLower1_26', 'rightEyeLower1_112', 'rightEyeLower1_243', 'rightEyeUpper2_113',
                           'rightEyeUpper2_225', 'rightEyeUpper2_224', 'rightEyeUpper2_223', 'rightEyeUpper2_222',
                           'rightEyeUpper2_221', 'rightEyeUpper2_189', 'rightEyeLower2_226', 'rightEyeLower2_31',
                           'rightEyeLower2_228', 'rightEyeLower2_229', 'rightEyeLower2_230', 'rightEyeLower2_231',
                           'rightEyeLower2_232', 'rightEyeLower2_233', 'rightEyeLower2_244', 'rightEyeLower3_143',
                           'rightEyeLower3_111', 'rightEyeLower3_117', 'rightEyeLower3_118', 'rightEyeLower3_119',
                           'rightEyeLower3_120', 'rightEyeLower3_121', 'rightEyeLower3_128', 'rightEyeLower3_245',
                           'rightEyebrowUpper_156', 'rightEyebrowUpper_70', 'rightEyebrowUpper_63',
                           'rightEyebrowUpper_105', 'rightEyebrowUpper_66', 'rightEyebrowUpper_107',
                           'rightEyebrowUpper_55', 'rightEyebrowUpper_193', 'rightEyebrowLower_35',
                           'rightEyebrowLower_124', 'rightEyebrowLower_46', 'rightEyebrowLower_53',
                           'rightEyebrowLower_52', 'rightEyebrowLower_65', 'leftEyeUpper0_466', 'leftEyeUpper0_388',
                           'leftEyeUpper0_387', 'leftEyeUpper0_386', 'leftEyeUpper0_385', 'leftEyeUpper0_384',
                           'leftEyeUpper0_398', 'leftEyeLower0_263', 'leftEyeLower0_249', 'leftEyeLower0_390',
                           'leftEyeLower0_373', 'leftEyeLower0_374', 'leftEyeLower0_380', 'leftEyeLower0_381',
                           'leftEyeLower0_382', 'leftEyeLower0_362', 'leftEyeUpper1_467', 'leftEyeUpper1_260',
                           'leftEyeUpper1_259', 'leftEyeUpper1_257', 'leftEyeUpper1_258', 'leftEyeUpper1_286',
                           'leftEyeUpper1_414', 'leftEyeLower1_359', 'leftEyeLower1_255', 'leftEyeLower1_339',
                           'leftEyeLower1_254', 'leftEyeLower1_253', 'leftEyeLower1_252', 'leftEyeLower1_256',
                           'leftEyeLower1_341', 'leftEyeLower1_463', 'leftEyeUpper2_342', 'leftEyeUpper2_445',
                           'leftEyeUpper2_444', 'leftEyeUpper2_443', 'leftEyeUpper2_442', 'leftEyeUpper2_441',
                           'leftEyeUpper2_413', 'leftEyeLower2_446', 'leftEyeLower2_261', 'leftEyeLower2_448',
                           'leftEyeLower2_449', 'leftEyeLower2_450', 'leftEyeLower2_451', 'leftEyeLower2_452',
                           'leftEyeLower2_453', 'leftEyeLower2_464', 'leftEyeLower3_372', 'leftEyeLower3_340',
                           'leftEyeLower3_346', 'leftEyeLower3_347', 'leftEyeLower3_348', 'leftEyeLower3_349',
                           'leftEyeLower3_350', 'leftEyeLower3_357', 'leftEyeLower3_465', 'leftEyebrowUpper_383',
                           'leftEyebrowUpper_300', 'leftEyebrowUpper_293', 'leftEyebrowUpper_334',
                           'leftEyebrowUpper_296', 'leftEyebrowUpper_336', 'leftEyebrowUpper_285',
                           'leftEyebrowUpper_417', 'leftEyebrowLower_265', 'leftEyebrowLower_353',
                           'leftEyebrowLower_276', 'leftEyebrowLower_283', 'leftEyebrowLower_282',
                           'leftEyebrowLower_295', 'midwayBetweenEyes_168', 'noseTip_1', 'noseBottom_2',
                           'noseRightCorner_98', 'noseLeftCorner_327', 'rightCheek_205', 'rightCheek_137',
                           'rightCheek_123', 'rightCheek_50', 'rightCheek_203', 'rightCheek_177', 'rightCheek_147',
                           'rightCheek_187', 'rightCheek_207', 'rightCheek_216', 'rightCheek_215', 'rightCheek_213',
                           'rightCheek_192', 'rightCheek_214', 'rightCheek_212', 'rightCheek_138', 'rightCheek_135',
                           'rightCheek_210', 'rightCheek_169', 'leftCheek_425', 'leftCheek_352', 'leftCheek_280',
                           'leftCheek_330', 'leftCheek_266', 'leftCheek_423', 'leftCheek_426', 'leftCheek_427',
                           'leftCheek_411', 'leftCheek_376', 'leftCheek_436', 'leftCheek_416', 'leftCheek_432',
                           'leftCheek_434', 'leftCheek_422', 'leftCheek_430', 'leftCheek_364', 'leftCheek_394',
                           'leftCheek_371'])

# Write the DataFrame to a CSV file
# csv_output_path = "Landmarks_Non_Stroke_Faces_withouts.csv"
csv_output_path = "Final_Non_Stroke.csv"
df.to_csv(csv_output_path, index=False)
