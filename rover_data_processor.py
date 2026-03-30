import os
import numpy as np
import cv2
import csv
import rosbag

NAME = "cc_m_10"
SOURCE_PATH = "rawdata/"
DEST_PATH = "processed/bw"
resize = (160, 120)


def load_telem_file(path):
    """
    Loads telemetry data from a CSV file.
    """
    with open(path, "r") as f:
        dict_reader = csv.DictReader(f)
        return list(dict_reader)


def process_bag_file(bag_file):
    """
    Extract images from a rosbag and convert them to binary black/white.
    Blur is applied to remove noise before thresholding.
    """
    img_topic = "/device_0/sensor_1/Color_0/image/data"

    output_folder = os.path.join(DEST_PATH, NAME)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Extract images from {bag_file} on topic {img_topic} into {output_folder}")

    bag = rosbag.Bag(bag_file, "r")

    count = 0
    count_real = 0

    frm_lookup = load_telem_file(bag_file.replace(".bag", ".csv"))

    for topic, msg, t in bag.read_messages(topics=[img_topic]):
        try:
            result = [entry for entry in frm_lookup if entry["index"] == str(count)]

            throttle = result[0]["throttle"]
            steering = result[0]["steering"]
            heading = result[0]["heading"]

            img_name = f"{int(count):05d}_{throttle}_{steering}_{heading}.png"
            file_path = os.path.join(output_folder, img_name)

            encoding = msg.encoding

            if encoding == "mono16":
                cv_img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
                normalized = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            else:
                cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                normalized = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Resize
            resized = cv2.resize(normalized, resize)

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

            # Blur to remove noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Threshold to binary
            _, bw = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

            # Save image
            cv2.imwrite(file_path, bw)

            if count % 10 == 0:
                print(f"Wrote image {count}")

            count_real += 1

        except Exception as e:
            print("Could not find matching features to the image. Skipping. Count =", count, e)

        count += 1

    print("Number of images:", count)
    print("Number of images processed:", count_real)

    bag.close()


def main():
    """
    Process all .bag files in the source directory.
    """
    source = os.path.join(SOURCE_PATH, NAME)

    for filename in os.listdir(source):
        if filename.endswith(".bag"):
            process_bag_file(os.path.join(source, filename))


if __name__ == "__main__":
    main()
