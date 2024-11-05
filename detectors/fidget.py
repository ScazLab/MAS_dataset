"""
This file defines the `FidgetNode` class, which connects MoveNet and the Person class to detect fidgeting.
"""

from utils.person import Person
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import os
import argparse
from termcolor import colored, cprint

class FidgetNode:
    """
    A class for detecting fidgeting behavior in video frames using the MoveNet pose estimation model.
    This node tracks the positions and movements of people to identify fidgeting based on body, hand, and face movements.
    """

    def __init__(self, matrix_memory_length: int = 20, json_record: bool = False, 
                 body_fidget_thresh: float = 0.3, body_optical_thresh: float = 0.8, 
                 hand_fidget_thresh: float = 0.3, hand_optical_thresh: float = 1.5,
                 face_optical_thresh: float = 0.2, action_memory_length: float = float('inf'), 
                 long_term_recorder: Optional['MatrixRecorder'] = None):
        """
        Initializes the FidgetNode with specified parameters for fidget detection.

        Args:
            matrix_memory_length (int): Number of frames of fidget data to store in memory.
            json_record (bool): Whether to save fidget matrices to a JSON file when memory is cleared.
            body_fidget_thresh (float): Threshold for body fidget detection based on movement.
            body_optical_thresh (float): Optical flow threshold for detecting body fidgets.
            hand_fidget_thresh (float): Threshold for hand fidget detection based on movement.
            hand_optical_thresh (float): Optical flow threshold for detecting hand fidgets.
            face_optical_thresh (float): Optical flow threshold for detecting face movements.
            action_memory_length (float): Number of actions to remember for determining fidgeting behavior.
            long_term_recorder (Optional[MatrixRecorder]): An optional recorder for long-term storage of fidget data.
        """
        with tf.device('/CPU:0'):
            self.lightning = [hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4"), 192]
            self.thunder = [hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4"), 256]

        self.matrix_memory_length = matrix_memory_length
        self.people: List[Person] = []  # List to store detected persons and their movements
        self.json_record = json_record
        self.matrices: List[np.ndarray] = []  # List to store fidget matrices
        self.body_optical_thresh = body_optical_thresh
        self.body_fidget_thresh = body_fidget_thresh
        self.hand_optical_thresh = hand_optical_thresh
        self.hand_fidget_thresh = hand_fidget_thresh
        self.face_optical_thresh = face_optical_thresh
        self.action_memory: List[int] = []  # List to store detected fidget actions
        self.action_memory_length = action_memory_length
        self.long_term_recorder = long_term_recorder

    def movenet(self, frame: np.ndarray, module, model_dims: int) -> np.ndarray:
        """
        Estimates poses in a given frame using the MoveNet model.

        Args:
            frame (np.ndarray): The input image frame.
            module: The MoveNet module for pose estimation.
            model_dims (int): The input image dimensions for the model.

        Returns:
            np.ndarray: An array of shape [17, 3] representing keypoint coordinates and scores.
        """
        with tf.device('/CPU:0'):
            model = module.signatures['serving_default']
            input_image = self.movenet_input(frame, model_dims)
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = model(input_image)
            keypoints_with_scores = outputs['output_0'].numpy()
            keypoints_with_scores = keypoints_with_scores[0][0]
            self.convert_keypoint_fractions(keypoints_with_scores, frame.shape[:2], input_image.shape[1:3])
            return keypoints_with_scores

    def movenet_input(self, frame: np.ndarray, input_size: int) -> tf.Tensor:
        """
        Prepares an image for input into the MoveNet model by resizing and padding.

        Args:
            frame (np.ndarray): The image to resize.
            input_size (int): The new size for the image.

        Returns:
            tf.Tensor: The resized and padded image tensor.
        """
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
        return input_image

    def convert_keypoint_fractions(self, keypoints: np.ndarray, dimensions: Tuple[int, int], model_dimensions: Tuple[int, int]):
        """
        Recalculates keypoint fractions to match the original image dimensions instead of the resized model input.

        Args:
            keypoints (np.ndarray): The raw keypoints generated by MoveNet.
            dimensions (Tuple[int, int]): The original image dimensions.
            model_dimensions (Tuple[int, int]): The dimensions used by the model.

        Returns:
            np.ndarray: A list of recalculated keypoints.
        """
        if dimensions[0] / model_dimensions[0] > dimensions[1] / model_dimensions[1]:
            large_axis = 0
            small_axis = 1
        else:
            large_axis = 1
            small_axis = 0

        scaled_small_dim = dimensions[small_axis] * model_dimensions[large_axis] / dimensions[large_axis]
        padding_fraction = (model_dimensions[small_axis] - scaled_small_dim) / (2 * model_dimensions[small_axis])
        small_fraction = scaled_small_dim / model_dimensions[small_axis]
        keypoints[:, small_axis] = (keypoints[:, small_axis] - padding_fraction) / small_fraction
        keypoints[:, small_axis] = [x if x <= 1 else 0.99 for x in keypoints[:, small_axis]]

    def get_keypoint_pix(self, keypoints: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Converts keypoints from fractional to pixel coordinates based on the image dimensions.

        Args:
            keypoints (np.ndarray): The fractional keypoints.
            frame (np.ndarray): The image to use for scaling.

        Returns:
            np.ndarray: Keypoints in pixel coordinates.
        """
        keypoints = keypoints.copy()
        keypoints[:, 0] *= frame.shape[0]
        keypoints[:, 1] *= frame.shape[1]
        return keypoints

    def paint_keypoints(self, points: np.ndarray, frame: np.ndarray, c: Tuple[int, int, int] = (0, 0, 255), display_confidence: bool = False) -> np.ndarray:
        """
        Paints keypoints onto an image.

        Args:
            points (np.ndarray): List of keypoints to paint.
            frame (np.ndarray): The image to paint keypoints on.
            c (Tuple[int, int, int]): Color for the keypoints.
            display_confidence (bool): Whether to display confidence values.

        Returns:
            np.ndarray: The annotated image.
        """
        confidence_threshold = 0.2
        for i, p in enumerate(points):
            if p[2] < confidence_threshold:
                continue
            p1 = (int(p[1]), int(p[0]))
            p2 = (p1[0] + 5, p1[1] + 5)
            frame = cv2.rectangle(frame, p1, p2, c, -1)
            if display_confidence:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                frame = cv2.putText(frame, str(p[2]), p1, font, font_scale, c, thickness, cv2.LINE_AA)
        return frame

    def find_people(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Uses MoveNet lightning to quickly identify the general location of people in the image.

        Args:
            frame (np.ndarray): The image to analyze.

        Returns:
            Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]: Points defining a rectangle around the person.
        """
        pose = self.movenet(frame, self.lightning[0], self.lightning[1])
        pose = self.get_keypoint_pix(pose, frame)

        conf_thresh = 0.2
        for i in reversed(range(len(pose))):
            if pose[i, 2] < conf_thresh:
                pose = np.delete(pose, i, 0)

        tolerance = 25

        if len(pose) == 0:
            return None, None

        min_point = [max(int(min(pose[:, 0]) - tolerance), 0), max(int(min(pose[:, 1]) - tolerance), 0)]
        max_point = [int(max(pose[:, 0]) + tolerance), int(max(pose[:, 1]) + tolerance)]
        if max_point[0] <= min_point[0]:
            max_point[0] = min_point[0] + 10

        if max_point[1] <= min_point[1]:
            max_point[1] = min_point[1] + 10

        min_point = tuple(min_point)
        max_point = tuple(max_point)
        return min_point, max_point

    def find_face(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Identifies faces in a frame using the MoveNet model.

        Args:
            frame (np.ndarray): The frame to search for faces.

        Returns:
            Optional[Tuple[Tuple[int, int], Tuple[int, int]]]: Points representing a bounding rectangle for the face.
        """
        pose = self.movenet(frame, self.lightning[0], self.lightning[1])
        pose = self.get_keypoint_pix(pose, frame)

        person = Person()
        person.update_skeleton(pose, 0, 0)
        return person.get_face_rect()

    def run_pose_estimation(self, clean_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs pose estimation on an image. First, identifies the location of people, then zooms in for a detailed analysis.

        Args:
            clean_frame (np.ndarray): The image to analyze.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Keypoints generated during pose estimation and the annotated image.
        """
        p_min, p_max = self.find_people(clean_frame)
        if p_min is None or p_max is None or p_max[0] - p_min[0] < 1 or p_max[1] - p_min[1] < 1 or p_min[0] < 0 or p_min[1] < 0 or p_max[0] < 0 or p_max[1] < 0:
            return np.array([]), clean_frame

        cropped_frame = clean_frame[p_min[0]:p_max[0], p_min[1]:p_max[1]]
        pose = self.movenet(cropped_frame, self.thunder[0], self.thunder[1])
        pose = self.get_keypoint_pix(pose, cropped_frame)
        pose[:, 0] += p_min[0]
        pose[:, 1] += p_min[1]

        frame = clean_frame
        return pose, frame

    def get_fidget_matrix(self, person: Person) -> np.ndarray:
        """
        Creates a matrix from the fidget data contained in a Person object.

        Args:
            person (Person): A Person object with detected limb movements.

        Returns:
            np.ndarray: A 2 x number_of_limbs x 2 matrix containing fidget data.
        """
        left_limbs = ['ruarm', 'rfarm', 'luleg', 'ruleg', 'llleg', 'rlleg', 'face', 'rhand']
        right_limbs = ['luarm', 'lfarm', 'luleg', 'ruleg', 'llleg', 'rlleg', 'face', 'lhand']
        fidget_matrix = np.zeros((2, len(left_limbs), 2))
        left_collisions, right_collisions = person.get_collisions()
        left_fidgets, right_fidgets = person.get_fidgets()

        fidget_matrix[0, :, 0] += [int(x in left_collisions) for x in left_limbs]
        fidget_matrix[0, :, 1] += [int(x in left_fidgets) for x in left_limbs]
        fidget_matrix[1, :, 0] += [int(x in right_collisions) for x in right_limbs]
        fidget_matrix[1, :, 1] += [int(x in right_fidgets) for x in right_limbs]

        return fidget_matrix

    def get_matrices(self) -> List[np.ndarray]:
        """
        Returns the matrices generated by get_fidget_matrix at every frame. If json_record is not enabled, the list will be empty.

        Returns:
            List[np.ndarray]: A list of matrices representing fidget data over time.
        """
        return self.matrices

    def get_avg_fidget_matrix(self) -> np.ndarray:
        """
        Computes the average fidget matrix over the frames in the node's memory.

        Returns:
            np.ndarray: A matrix representing the average fidget values per frame.
        """
        sum_matrix = np.mean(np.array(self.matrices), axis=0)
        return sum_matrix

    def get_fidget_action(self) -> int:
        """
        Returns the most recently calculated fidget action.

        Returns:
            int: The index of the most recent fidget action.
        """
        return self.action_memory[-1]

    def get_fidget_percentage(self, filter: List[int] = [1, 2, 3]) -> float:
        """
        Computes the percentage of frames where fidgeting actions were detected.

        Args:
            filter (List[int]): A list of fidget action indices to consider.

        Returns:
            float: The percentage of frames with detected fidgeting.
        """
        if len(self.action_memory) == 0:
            return 0.0

        avg = sum(1 if x in filter else 0 for x in self.action_memory) / len(self.action_memory)
        return avg

    def calc_fidget_action(self) -> int:
        """
        Determines the current fidget action based on the average fidget matrix.

        Returns:
            int: The index of the detected fidget action.
        """
        avg_matrix = self.get_avg_fidget_matrix()
        if avg_matrix[:, -1, 1].max() > self.hand_fidget_thresh:
            return 3
        elif avg_matrix[:, -2, 1].max() > self.body_fidget_thresh:
            return 1
        elif avg_matrix[:, :2, 1].max() > self.body_fidget_thresh:
            return 2

        return 0

    def clear_memory(self) -> None:
        """
        Clears the node's memory of fidget data. If json_record is enabled, saves the current matrices to a JSON file before clearing.
        """
        if self.json_record:
            curr_time = datetime.now().strftime("%m.%d.%Y,%H-%M-%S")
            with open(f'fidget_output_{curr_time}.json', 'w') as outfile:
                json.dump(np.array(self.matrices).tolist(), outfile)
        self.matrices = []
        self.people = []
        self.action_memory = []

    def set_thresholds(self, body_optical_thresh: float, body_fidget_thresh: float, 
                       hand_optical_thresh: float, hand_fidget_thresh: float) -> None:
        """
        Sets the thresholds for detecting fidgeting movements.

        Args:
            body_optical_thresh (float): Optical flow threshold for body fidgets.
            body_fidget_thresh (float): Fidget threshold for body movements.
            hand_optical_thresh (float): Optical flow threshold for hand fidgets.
            hand_fidget_thresh (float): Fidget threshold for hand movements.
        """
        self.body_optical_thresh = body_optical_thresh
        self.body_fidget_thresh = body_fidget_thresh
        self.hand_optical_thresh = hand_optical_thresh
        self.hand_fidget_thresh = hand_fidget_thresh

    def set_mem_length(self, length: int) -> None:
        """
        Sets the memory length for storing fidget data. Discards old data if the new length is shorter.

        Args:
            length (int): The new memory length.
        """
        self.matrix_memory_length = length
        while len(self.matrices) > self.matrix_memory_length:
            self.matrices.pop(0)

        while len(self.people) > self.matrix_memory_length:
            self.people.pop(0)

    def set_recorder(self, recorder: 'MatrixRecorder') -> None:
        """
        Sets a long-term recorder for storing fidget data.

        Args:
            recorder (MatrixRecorder): The recorder object to use for long-term storage.
        """
        self.long_term_recorder = recorder

    def detect_fidget(self, clean_frame: np.ndarray, prev_frame: np.ndarray, draw_collisions: bool = False) -> Tuple[Person, np.ndarray]:
        """
        Detects fidgeting in a given frame by comparing with a previous frame.

        Args:
            clean_frame (np.ndarray): The current clean frame (no annotations).
            prev_frame (np.ndarray): The previous clean frame.
            draw_collisions (bool): Whether to draw detected collisions on the frame.

        Returns:
            Tuple[Person, np.ndarray]: A tuple containing the detected person and the annotated frame.
        """
        pose, frame = self.run_pose_estimation(clean_frame)
        person = Person(body_optical_thresh=self.body_optical_thresh, hand_optical_thresh=self.hand_optical_thresh, face_optical_thresh=self.face_optical_thresh)
        person.update_skeleton(pose)

        collisions, fidgets = person.check_collisions(), person.check_fidgets(clean_frame, prev_frame)
        collisions, fidgets = collisions[0] + collisions[1], fidgets[0] + fidgets[1]

        if draw_collisions:
            if person.left_hand is not None:
                frame = person.left_hand.paint_quadrilateral(frame, c=(255, 0, 0))
            if person.right_hand is not None:
                frame = person.right_hand.paint_quadrilateral(frame, c=(255, 0, 0))

            for c in collisions:
                frame = person.limbs[c].paint_quadrilateral(frame, c=(0, 0, 255))

            for f in fidgets:
                frame = person.limbs[f].paint_quadrilateral(frame, c=(0, 255, 0))

        self.people.append(person)
        if len(self.people) > self.matrix_memory_length:
            self.people.pop(0)

        new_fidget_matrix = self.get_fidget_matrix(person)
        self.matrices.append(new_fidget_matrix)
        if len(self.matrices) > self.matrix_memory_length:
            self.matrices.pop(0)

        if self.long_term_recorder:
            self.long_term_recorder.record(new_fidget_matrix)

        self.action_memory.append(self.calc_fidget_action())
        if len(self.action_memory) > self.action_memory_length:
            self.action_memory.pop(0)

        return person, frame

    def ros_analyze(self, video_path: str) -> float:
        """
        Analyzes a sequence of frames to detect fidgeting behavior.

        Args:
            video_path (str): Path to the video file.

        Returns:
            float: The percentage of frames with detected fidgeting.
        """
        self.clear_memory()

        video = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not video.isOpened():
            print("Error: Could not open video.")

        frames = []
        while True:
            # Read the next frame from the video
            ret, frame = video.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                break

            # Add the frame to the list
            frames.append(frame)

        # Release the video capture object
        video.release()

        prev_frame = frames[0]
        for frame in frames[1:]:
            try:
                self.detect_fidget(frame, prev_frame, draw_collisions=False)
            except Exception as e:
                print(f"Error during fidget detection: {e}")
            prev_frame = frame

        return self.get_fidget_percentage()


    def ros_analyze(self, video_path: str) -> float:
        """
        Analyzes a sequence of frames to detect fidgeting behavior.

        Args:
            video_path (str): Path to the video file.

        Returns:
            float: The percentage of frames with detected fidgeting.
        """
        self.clear_memory()

        video = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not video.isOpened():
            print("Error: Could not open video.")

        frames = []
        while True:
            # Read the next frame from the video
            ret, frame = video.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                break

            # Add the frame to the list
            frames.append(frame)

        # Release the video capture object
        video.release()

        prev_frame = frames[0]
        for frame in frames[1:]:
            try:
                self.detect_fidget(frame, prev_frame, draw_collisions=False)
            except Exception as e:
                print(f"Error during fidget detection: {e}")
            prev_frame = frame

        return self.get_fidget_percentage()

class MatrixRecorder:
    """
    A class for recording and managing fidget matrices over a sequence of frames.
    """

    def __init__(self):
        """
        Initializes the MatrixRecorder with an empty memory.
        """
        self.memory: List[np.ndarray] = []

    def record(self, matrix: np.ndarray) -> None:
        """
        Records a fidget matrix.

        Args:
            matrix (np.ndarray): The matrix to record.
        """
        self.memory.append(matrix)

    def get_memory_avg(self) -> np.ndarray:
        """
        Computes the average fidget matrix over the stored matrices in memory.

        Returns:
            np.ndarray: The average fidget matrix.
        """
        return np.mean(np.array(self.memory), axis=0)

    def reset(self) -> None:
        """
        Resets the memory of recorded matrices.
        """
        self.memory = []

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Face Node")
    parser.add_argument('path', type=str, help='Path to the video file.')
    args = parser.parse_args()

    cprint('Initializing Fidget Node...', 'green', attrs=['bold'])
    node=FidgetNode()

    cprint('Analyzing Data...', 'green', attrs=['bold'])
    print(node.ros_analyze(args.path))