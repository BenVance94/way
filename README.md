# way
Who Are You (WAY)

## Overview
WAY is a Python-based tool designed to detect and analyze fake IDs. It leverages advanced image processing and OCR techniques to identify potential fraud and validate ID authenticity.

## Features

- **Image Analysis**:
  - Detects fake IDs using multiple indicators
  - Analyzes image quality and text placement
  - Identifies potential fraud based on image metrics

- **OCR Validation**:

    - Validates text against official ID templates
    - Checks for text consistency and placement
    - Verifies microprint and security features

- **Output**:

The system will output:
- Fraud score (0-100, where higher scores indicate potential fraud)
- Risk level (Low/Medium/High)
- Component scores for text and image analysis
- Quality metrics
- Detected fake indicators
- Extracted text


## Installation

1. Clone the repository:


    ```bash
    git clone https://github.com/yourusername/WAY.git
    cd WAY
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the tool:

    ```bash
    python run.py
    ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.