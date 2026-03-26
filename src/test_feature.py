__test__ = False

import pprint
from pathlib import Path
print("TEST SCRIPT STARTED")

def main():
    import cv2
    from feature_extraction import FeatureExtractor

    # ---------- CHANGE THIS ----------
    print("I am inside main now")
    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "Example Images" / "Test Images.jpg"   # put any face image here
    # ---------------------------------

    image = cv2.imread(str(image_path))
    print("Image shape:", None if image is None else image.shape)


    # if image is None:
    #     raise FileNotFoundError(f"Could not load image: {image_path}")
    # # print("FeatureExtractor imported from:", FeatureExtractor)

    extractor = FeatureExtractor()
    # print("FeatureExtractor imported from:", FeatureExtractor)
    print("Extractor initialized:", extractor)
    features = extractor.extract_features(image)

    print("\n===== EXTRACTED FEATURES =====")
    pprint.pprint(features)
    
    annotated = extractor.draw_annotations(image, features)

    cv2.imshow("Feature Extractor Output", annotated)
    print("\nPress any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
