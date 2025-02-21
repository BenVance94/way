print("Starting...")
from IDway import IDway
import os

# def get_images_from_directory(directory):
#     image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
#     image_files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
#     return image_files

# directory_path = 'dl_images/'
# images = get_images_from_directory(directory_path)

# if images:
#     print("Images found:")
#     for image in images:
#         print(image)
# else:
#     print("No images found in the directory.")

image_path = "dl_images/fake_id.jpg"
myWay = IDway(image_path)
print(myWay.output())

print("Done")