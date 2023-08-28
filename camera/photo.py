# photo.py
import os

def capture_photo(filename):
    command = f"libcamera-still -o {filename}"
    os.system(command)

if __name__ == "__main__":
    capture_photo("my_photo.jpg")
