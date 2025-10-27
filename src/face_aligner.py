"""Face alignment using dlib facial landmarks."""

from typing import Tuple
import numpy as np
import cv2
from imutils.face_utils.helpers import FACIAL_LANDMARKS_68_IDXS, FACIAL_LANDMARKS_5_IDXS, shape_to_np
import dlib


class FaceAligner:
    """Align faces based on eye positions using facial landmarks.

    This class handles face alignment by:
    1. Detecting facial landmarks (68-point or 5-point)
    2. Computing eye centers
    3. Rotating and scaling the face to align eyes to desired positions
    """

    def __init__(
        self,
        predictor: dlib.shape_predictor,
        desiredLeftEye: Tuple[float, float] = (0.35, 0.35),
        desiredFaceWidth: int = 256,
        desiredFaceHeight: int = None
    ):
        """Initialize the face aligner.

        Args:
            predictor: dlib shape predictor for facial landmarks
            desiredLeftEye: Desired (x, y) position of left eye as fraction of face width/height
            desiredFaceWidth: Target width of aligned face in pixels
            desiredFaceHeight: Target height of aligned face (defaults to desiredFaceWidth)
        """
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight or desiredFaceWidth

    def align(self, image: np.ndarray, gray: np.ndarray, rect: dlib.rectangle) -> np.ndarray:
        """Align a face in the image based on facial landmarks.

        Args:
            image: Original BGR image
            gray: Grayscale version of the image
            rect: dlib rectangle indicating face location

        Returns:
            Aligned face image as numpy array
        """
        # Convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # Determine which landmark indices to use based on shape points
        if len(shape) == 68:
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # Compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # Compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # Determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # Compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (
            int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        )

        # Grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # Update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # Apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return output

    def align_with_manual_landmarks(
        self,
        image: np.ndarray,
        left_eye_center: Tuple[int, int],
        right_eye_center: Tuple[int, int]
    ) -> np.ndarray:
        """Align a face using manually specified eye centers.

        Args:
            image: Original BGR image
            left_eye_center: (x, y) coordinates of left eye center
            right_eye_center: (x, y) coordinates of right eye center

        Returns:
            Aligned face image as numpy array
        """
        # Convert to numpy arrays
        leftEyeCenter = np.array(left_eye_center, dtype="int")
        rightEyeCenter = np.array(right_eye_center, dtype="int")

        # Compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Compute the desired right eye x-coordinate
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # Determine the scale
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # Compute center between the two eyes
        eyesCenter = (
            int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        )

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # Update the translation component
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # Apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return output
