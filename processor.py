import cv2
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from functions import resize_pad, extract_features


# Data and Label containers
features = []
labels = []


for folder in range(10):
    # Load and preprocess each image
    for file in range(1,12):
        image_path = f"./{folder}/sample{file}.png"
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            preprocessed_image = resize_pad(image)
            features.append(extract_features(preprocessed_image))
            labels.append(folder)  # Label is the digit folder number (0-9)
            print(image_path)
            cv2.imshow(image_path, preprocessed_image)
            cv2.imwrite(f"{folder}_{file}.png", preprocessed_image)
            cv2.waitKey(50)
            cv2.destroyAllWindows()
        except Exception as e:
            continue



# Convert data to NumPy arrays for easier handling
features = np.array(features)
labels = np.array(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM model (using linear kernel here, experiment with others)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Save the model using joblib.dump
joblib.dump(clf, 'svm_model.pkl')

print("Model saved to svm_model.pkl")

# Prediction on a new unseen image
# Replace 'new_image.png' with your actual test image path
# new_image = cv2.imread('sample10.png', cv2.IMREAD_GRAYSCALE)
# new_image_processed = resize_pad(new_image)
# new_features = extract_features(new_image_processed)
# predicted_label = clf.predict(np.array([new_features]))

# print(f"Predicted digit for new image: {predicted_label[0]}")

# Further evaluation (accuracy etc.) on the test set (y_test, X_test)