"""
This file defines the `FaceNode` class, which is used for facial emotion recognition.
"""


import os
from fer import FER
from fer.utils import draw_annotations
import torch
import cv2
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import argparse
from termcolor import colored, cprint
import time
from fidget import FidgetNode


class FaceNode:
    """
    A class for detecting faces and recognizing emotions in images using the FER (Facial Emotion Recognition) library.
    This node stores the emotions detected over a sequence of frames and can compute average emotion scores.
    """

    def __init__(self, memory_length: int = 10):
        """
        Initializes the FaceNode with a specified memory length for storing detected faces and their emotions.

        Args:
            memory_length (int): Number of frames of face data to remember.
        """
        self.detector = FER(mtcnn=True)  # Initialize the facial emotion detector with MTCNN
        self.memory_length = memory_length
        self.faces: List[List[Dict[str, float]]] = []  # List to store detected faces and their emotions

    def recognize_face(self, frame: cv2.Mat) -> List[Dict[str, Dict[str, float]]]:
        """
        Runs FER on an image to identify faces and determine the emotions they are expressing.

        Args:
            frame (cv2.Mat): Image to analyze for faces.

        Returns:
            List[Dict[str, Dict[str, float]]]: A list of dictionaries containing detected faces and their emotion scores.
                                               Each dictionary represents a face with its bounding box and emotions.
                                               The 'emotions' key holds another dictionary with emotion labels and scores.
        """
        if frame is None or frame.size == 0:
            return []

        # Detect faces and their emotions in the frame
        faces = self.detector.detect_emotions(frame)
        
        # Append detected faces and emotions to memory
        self.faces.append(faces)
        if len(self.faces) > self.memory_length:
            self.faces.pop(0)
        
        return faces

    def get_avg_emotions(self) -> Dict[str, float]:
        """
        Computes the average emotion scores over the stored frames in the node's memory.

        Returns:
            Dict[str, float]: A dictionary containing each emotion and their average scores over the node's memory.
                              The dictionary keys are emotion labels (e.g., "happy", "sad"), and the values are averaged scores.
        """
        emo_dic: Dict[str, float] = {}
        mean_denominator = 0

        for frame in self.faces:
            for face in frame:
                mean_denominator += 1
                for emotion, score in face['emotions'].items():
                    if emotion not in emo_dic:
                        emo_dic[emotion] = 0
                    emo_dic[emotion] += score

        # Calculate average scores
        for emotion in emo_dic:
            emo_dic[emotion] /= mean_denominator
        
        return emo_dic

    def clear_memory(self) -> None:
        """
        Clears the node's memory of detected faces and emotions.
        """
        self.faces = []

    def analyze(self, video_path: str) -> Tuple[Dict[str, float], float]:
        """
        Analyzes a sequence of frames for facial emotions and calculates the percentage of frames with no detected faces.

        Args:
            video_path (str): Path to the video file.

        Returns:
            Tuple[Dict[str, float], float]: A dictionary of average emotion scores and the percentage of frames without detected faces.
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

        empty_frames = 0
        
        for frame in frames:
            try:
                self.recognize_face(frame)
                if not self.faces[-1]:  # Check if no faces were detected in the last frame
                    empty_frames += 1
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        avg_emotions = self.get_avg_emotions()
        off_screen_percent = empty_frames / len(frames) if frames else 0
        
        return avg_emotions, off_screen_percent


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Face Node")
    parser.add_argument('path', type=str, help='Path to the video file.')
    args=parser.parse_args()

    cprint('Initializing Face Node...', 'green', attrs=['bold'])
    node=FaceNode()

    cprint('Analyzing Data...', 'green', attrs=['bold'])
    print(node.analyze(args.path))


