print("Starting...")
from IDway import IDway
import os
import json

# Read user information from input.json
try:
    with open('input.json', 'r') as f:
        user_info = json.load(f)
except FileNotFoundError:
    print("Error: input.json file not found")
    exit(1)
except json.JSONDecodeError:
    print("Error: input.json is not valid JSON")
    exit(1)

# Get all image files from the dl_images directory
image_dir = "dl_images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
for image_file in image_files:
    print("\nRunning IDway on: ", image_file)
    print("--------------------------------")
    
    # Get user information for this image from the JSON file
    image_info = user_info.get(image_file)
    
    if not image_info:
        print(f"Warning: No information found for {image_file} in input.json")
        continue
    
    # Extract required fields with error checking
    required_fields = ['first_name', 'last_name', 'street_address', 'date_of_birth']
    missing_fields = [field for field in required_fields if field not in image_info]
    
    if missing_fields:
        print(f"Error: Missing required fields for {image_file}: {', '.join(missing_fields)}")
        continue
    
    image_path = os.path.join(image_dir, image_file)
    print(f"\nProcessing: {image_path}")
    print(f"Using information from input.json for {image_file}")
    
    myWay = IDway(
        image_path,
        first_name=image_info['first_name'],
        last_name=image_info['last_name'],
        street_address=image_info['street_address'],
        date_of_birth=image_info['date_of_birth']
    )
    myJson = myWay.output()
    print(myJson)

print("\nDone")