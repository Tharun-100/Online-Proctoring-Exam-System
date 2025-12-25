import cv2
import pprint
from feature_extraction import FeatureExtractor  # adjust filename if needed
print("TEST SCRIPT STARTED")

def main():
    # ---------- CHANGE THIS ----------
    print("I am inside main now")
    image_path = r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\Example Images\Test Images.jpg"   # put any face image here
    # ---------------------------------

    image = cv2.imread(image_path)
    print("Image shape:", None if image is None else image.shape)


    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    # print("FeatureExtractor imported from:", FeatureExtractor)

    extractor = FeatureExtractor()
    print("FeatureExtractor imported from:", FeatureExtractor)
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
