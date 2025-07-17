# python3 json2png.py --json_path "/mnt/lslteam/SIIRI/CodePythonSIIRI/HSI_Datasets_LSL_(Amine)/DataSets/PushBroom/UseCase_1_(Avoine1)/Anomaly_Easy/test/Annotation/JSON/test_UseCase_1_(Avoine1)_Anomaly_Easy_L1_1.json" --output_dir "/mnt/lslteam/SIIRI/CodePythonSIIRI/HSI_Datasets_LSL_(Amine)/DataSets/PushBroom/UseCase_1_(Avoine1)/Anomaly_Easy/test/Annotation/PNG"
# python3 json2png.py --json_path "/mnt/lslteam/SIIRI/CodePythonSIIRI/HSI_Datasets_LSL_(Amine)/DataSets/PushBroom/UseCase_1_(Avoine1)/Anomaly_Easy/train/Annotation/JSON/train_UseCase_1_(Avoine1)_Anomaly_Easy.json" --output_dir "/dataset_patching"

# python3 json2png.py --json_path "/mnt/lslteam/SIIRI/CodePythonSIIRI/HSI_Datasets_LSL_(Amine)/DataSets/PushBroom/UseCase_1_(Avoine1)/Anomaly_Hard/test/Annotation/JSON/test_UseCase_1_(Avoine1)_Anomaly_Hard.json" --output_dir "/mnt/lslteam/SIIRI/CodePythonSIIRI/HSI_Datasets_LSL_(Amine)/DataSets/PushBroom/UseCase_1_(Avoine1)/Anomaly_Hard/test/Annotation/PNG"



import json
import os
import argparse
import re
from PIL import Image, ImageDraw

def modify_name(name):
    # Supprimer le préfixe avant "UseCase_"
    modified_name = re.sub(r"^[\w-]+-", "", name)  # Supprime tout ce qui précède "UseCase_"
    
    # Utiliser une expression régulière pour trouver des mots entre parenthèses et les entourer de parenthèses si nécessaire
    modified_name = re.sub(r"(UseCase_\d+)_([A-Za-z0-9]+)(?=_)", r"\1_(\2)", modified_name)
    
    return modified_name



def create_mask_from_annotations(json_path, output_dir):
    # Load the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over the annotations in the JSON file
    for item in data:
        # Extract the image name
        image_name = item.get("image", f"image_{item.get('id', 'unknown')}")  # Path of the image
        image_base_name = os.path.splitext(os.path.basename(image_name))[0]  # Get the base name without extension
        image_base_name = modify_name(image_base_name)
        
        labels = item.get("label", [])
        
        if not labels:
            print(f"No annotations found for {image_name}. Skipping...")
            continue  # Skip if no annotations
        
        # Get the original image dimensions
        original_width = labels[0].get("original_width")
        original_height = labels[0].get("original_height")
        if not (original_width and original_height):
            print(f"Missing dimensions for {image_name}. Skipping...")
            continue  # Skip if dimensions are missing
        
        # Create a blank mask image (black background)
        mask = Image.new('L', (original_width, original_height), 0)  # 'L' mode for grayscale
        draw = ImageDraw.Draw(mask)
        
        for label in labels:
            points = label.get("points", [])
            if points:
                # Convert points from percentages to pixel coordinates
                polygon = [(x * original_width / 100, y * original_height / 100) for x, y in points]
                draw.polygon(polygon, outline=255, fill=255)  # Draw the polygon
        
        # Save the mask as a PNG file with the same name as the RGB image
        mask_filename = os.path.join(output_dir, f"{image_base_name}.png")
        mask.save(mask_filename)
        print(f"Mask saved: {mask_filename}")

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Generate PNG masks from JSON annotations.")
    parser.add_argument('--json_path', required=True, help="Path to the JSON file containing annotations.")
    parser.add_argument('--output_dir', required=True, help="Path to the directory where masks will be saved.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function
    create_mask_from_annotations(args.json_path, args.output_dir)

if __name__ == "__main__":
    main()
