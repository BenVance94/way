import cv2
import pytesseract
from PIL import Image
import re

class IDway:
    def __init__(self, image_path):
        self.image_path = image_path

    def preprocess_image(image_path):
        """
        Preprocess the image for better OCR accuracy.
        """
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get a binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def extract_text_from_image(image):
        """
        Extract text from the preprocessed image using Tesseract OCR.
        """
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(image)
        return text

    def validate_dl_text(text):
        """
        Validate the extracted text to check if it contains key DL fields.
        """
        # Define regex patterns for key fields
        patterns = {
            "name": r"[A-Z]+[A-Z]+,\s[A-Z]",  # Simple name pattern
            "date_of_birth": r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # Date of birth (MM/DD/YYYY)
            "license_number": r"[A-Z0-9]{6,}",  # License number (alphanumeric)
        }
        
        # Check for the presence of each pattern
        validation_results = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, text)
            validation_results[field] = bool(match)
        
        return validation_results

    def validate_dl(image_path):
        """
        Main function to process the DL image and validate it.
        """
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Extract text from the image
        extracted_text = extract_text_from_image(processed_image)
        print("Extracted Text:\n", extracted_text)
        
        # Validate the extracted text
        validation_results = validate_dl_text(extracted_text)
        print("\nValidation Results:")
        for field, result in validation_results.items():
            print(f"{field}: {'Valid' if result else 'Invalid'}")
        
        # Determine if the DL is likely legit or fake
        if all(validation_results.values()):
            print("\nConclusion: The DL appears to be legit.")
        else:
            print("\nConclusion: The DL might be fake or incomplete.")