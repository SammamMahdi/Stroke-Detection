import pygame
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import time
import pickle

# Load the pre-trained Voting Classifier model
with open('Voting_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Mediapipe Face Mesh
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.7)

# Define colors and fonts
BACKGROUND_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
FONT_NAME = "Arial"
FONT_SIZE = 24
BUTTON_FONT_SIZE = 20

# Define UI dimensions
WIDTH = 800
HEIGHT = 600

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stroke Detection System")
font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
button_font = pygame.font.SysFont(FONT_NAME, BUTTON_FONT_SIZE)
clock = pygame.time.Clock()


# Define button class
class Button:
    def __init__(self, text, x, y, width, height, color, hover_color, font):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.hover_color = hover_color
        self.font = font

    def draw(self, screen, mouse_pos):
        current_color = self.hover_color if self.is_hovered(mouse_pos) else self.color
        pygame.draw.rect(screen, current_color, (self.x, self.y, self.width, self.height))
        text_surface = self.font.render(self.text, True, TEXT_COLOR)
        screen.blit(text_surface, (
        self.x + (self.width - text_surface.get_width()) // 2, self.y + (self.height - text_surface.get_height()) // 2))

    def is_hovered(self, mouse_pos):
        return self.x <= mouse_pos[0] <= self.x + self.width and self.y <= mouse_pos[1] <= self.y + self.height


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


# Button instances
upload_button = Button("Upload Image", WIDTH // 2 - 100, HEIGHT // 2 - 30, 200, 50, BUTTON_COLOR, BUTTON_HOVER_COLOR,
                       button_font)


# Main function
def main():
    running = True
    result_text = ""
    while running:
        screen.fill(BACKGROUND_COLOR)
        mouse_pos = pygame.mouse.get_pos()

        # Draw title
        title_surface = font.render("Stroke Detection System", True, TEXT_COLOR)
        screen.blit(title_surface, ((WIDTH - title_surface.get_width()) // 2, 50))

        # Draw button
        upload_button.draw(screen, mouse_pos)

        # Draw result text
        result_surface = font.render(result_text, True, TEXT_COLOR)
        screen.blit(result_surface, ((WIDTH - result_surface.get_width()) // 2, HEIGHT - 100))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if upload_button.is_hovered(mouse_pos):
                    file_path = open_file_dialog()
                    if file_path:
                        result_text = process_image(file_path)

        pygame.display.flip()
        clock.tick(30)


# Function to open file dialog
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    return file_path


# Function to process image
def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return "Error loading image."

    total_start_time = time.time()

    # Extract landmarks
    start_preprocessing = time.time()
    features = get_landmarks(image)
    preprocessing_time = time.time() - start_preprocessing

    if features and len(features) == 115:
        features = np.array(features).reshape(1, -1)

        # Perform inference
        start_inference = time.time()
        prediction = model.predict(features)
        inference_time = time.time() - start_inference

        label = "Stroke" if prediction[0] == 1 else "Non-Stroke"
        total_time = time.time() - total_start_time

        return f"Label: {label} | Preprocessing: {preprocessing_time * 1000:.2f}ms | Inference: {inference_time * 1000:.2f}ms | Total: {total_time * 1000:.2f}ms"
    else:
        return "Feature extraction failed or insufficient features."


if __name__ == "__main__":
    main()
    pygame.quit()
