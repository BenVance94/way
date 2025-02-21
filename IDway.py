import cv2
import pytesseract
from PIL import Image
import re

class IDway:
    def __init__(self, image_path):
        self.image_path = image_path

    def _preprocess_image(self):
        '''
        Converts to Gray Scale and then Binary (using CV2)
        '''
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR) # Load the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) # Apply thresholding to get a binary image
        return binary
    
    def _extract_text_from_image(self, image):
        '''
        Use Tesseract to extract text from the altered image
        '''
        return pytesseract.image_to_string(image) 

    def _validate_dl_text(self, text):
        '''
        Using Regex to match on the text extracted from the altered image
        - Name pattern
        - Date of Birth
        - License Number
        '''
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

    def output(self):
        '''
        Runs all the methods and produces an output
        '''
        processed_image = self._preprocess_image() # Preprocess the image
        extracted_text = self._extract_text_from_image(processed_image) # Extract text from the image
        validation_results = self._validate_dl_text(extracted_text) # Validate the extracted text

        #print("Extracted Text:\n", extracted_text)
        print("\nValidation Results:")
        for field, result in validation_results.items():
            print(f"{field}: {'Valid' if result else 'Invalid'}")
        
        # Determine if the DL is likely legit or fake
        if all(validation_results.values()):
            print("\nConclusion: The DL appears to be legit.")
        else:
            print("\nConclusion: The DL might be fake or incomplete.")