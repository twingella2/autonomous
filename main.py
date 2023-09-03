from image_analysis.object_detection import main as detect_objects_main
from image_analysis.line_tracking import check_line_tracking_sensor
from image_analysis.ultrasonic_sensor import check_ultrasonic_sensor


def main():
    print("Starting object detection...")
    detect_objects_main()
    print("Object detection completed.")
    check_line_tracking_sensor()
    check_ultrasonic_sensor()

if __name__ == "__main__":
    main()
