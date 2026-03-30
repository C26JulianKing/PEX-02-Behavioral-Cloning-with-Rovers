"""
rover_driver.py
"""

import pyrealsense2.pyrealsense2 as rs
import time
import numpy as np
import cv2
import keras
import utilities.drone_lib as dl

# Path to the trained model weights
MODEL_NAME = "models/V6.6_SGD_Batch115_664Steps/rover_model06_ver06_epoch0095_val_loss0.005095.h5"

# Rover driving command limits
MIN_STEERING, MAX_STEERING = 1000, 2000
MIN_THROTTLE, MAX_THROTTLE = 1500, 2000

"""
HINT:  Get values to the above by querying your own rover...
rover.parameters['RC3_MAX']
rover.parameters['RC3_MIN']
rover.parameters['RC1_MAX']
rover.parameters['RC1_MIN']
"""

# Image processing parameters
white_L, white_H = 200, 255  # kept for reference
resize_W, resize_H = 160, 120  # must match training image generation
crop_W, crop_B, crop_T = 160, 120, 40  # currently unused


def get_model(filename):
    """Load the Keras model."""
    model = keras.models.load_model(filename, compile=False)
    print("Loaded Model")
    return model


def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)


def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min


def denormalize(steering, throttle):
    """
    Convert normalized model outputs back to rover command values.

    Training normalized labels using y_min=1000 and y_max=2000 for BOTH outputs.
    So we must invert with the same range used during training.
    """
    steering = invert_min_max_norm(steering, 1000.0, 2000.0)
    throttle = invert_min_max_norm(throttle, 1000.0, 2000.0)
    return steering, throttle


def initialize_pipeline(brg=False):
    """Initialize the RealSense pipeline for video capture."""
    pipeline = rs.pipeline()
    config = rs.config()

    if brg:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    pipeline.start(config)
    return pipeline


def get_video_data(pipeline):
    """
    Capture a frame and preprocess it exactly like the training image script:
    1. normalize to 0-255
    2. resize to (160, 120)
    3. convert RGB to grayscale
    4. Gaussian blur
    5. threshold at 200
    6. add channel dim and batch dim for CNN
    """
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    if not color_frame:
        return None

    image = np.asanyarray(color_frame.get_data())

    # Match bag processor behavior as closely as possible
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Resize exactly like training preprocessing script
    resized = cv2.resize(normalized, (resize_W, resize_H))

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to binary
    _, bw = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # Convert to float32 for model input
    bw = bw.astype(np.float32)

    # Add channel dimension: (120, 160, 1)
    bw = np.expand_dims(bw, axis=-1)

    # Add batch dimension: (1, 120, 160, 1)
    bw = np.expand_dims(bw, axis=0)

    return bw


def set_rover_data(rover, steering, throttle):
    """Set rover control commands."""
    rover.channels.overrides = {"1": steering, "3": throttle}
    print(f"Steering: {steering}, Throttle: {throttle}")


def check_inputs(steering, throttle):
    """Clamp commands to valid rover ranges."""
    steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
    throttle = np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle


def main():
    """Main inference loop."""
    rover = dl.connect_device("/dev/ttyACM0")
    model = get_model(MODEL_NAME)

    if model is None:
        print("Unable to load CNN model!")
        rover.close()
        print("Terminating program...")
        return

    while True:
        print("Arm vehicle to start mission.")
        print("(CTRL-C to stop process)")

        while not rover.armed:
            time.sleep(1)

        pipeline = initialize_pipeline()

        try:
            while rover.armed:
                processed_image = get_video_data(pipeline)
                if processed_image is None:
                    print("No image from camera.")
                    continue

                # Predict normalized outputs
                # Output order from training is [steering, throttle]
                output = model.predict(processed_image, verbose=0)

                pred_steering_norm = float(output[0][0])
                pred_throttle_norm = float(output[0][1])

                # Convert normalized outputs back to raw PWM-like values
                steering, throttle = denormalize(pred_steering_norm, pred_throttle_norm)

                # Convert to ints and clamp to valid rover command ranges
                steering, throttle = check_inputs(int(round(steering)), int(round(throttle)))

                # Send commands to rover
                set_rover_data(rover, steering, throttle)

        finally:
            pipeline.stop()
            time.sleep(1)
            rover.close()
            print("Done.")


if __name__ == "__main__":
    main()
