import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage import io
from scipy import ndimage
from lbp import LBP
from getmapping import getmapping


def load_images(image_folder):
    images = []
    labels = []
    categories = [d for d in os.listdir(
        image_folder) if os.path.isdir(os.path.join(image_folder, d))]

    for category in categories:
        category_path = os.path.join(image_folder, category)
        for image_file in os.listdir(category_path):
            if image_file.endswith('.png'):
                image_path = os.path.join(category_path, image_file)
                image = io.imread(image_path, as_gray=True)
                if image.ndim > 2:
                    # Convert to grayscale if it's not already
                    image = np.mean(image, axis=2)
                images.append(image)
                labels.append(category)

    return images, np.array(labels), categories


def extract_lbp_features(images, mapping):
    feature_matrix = []
    for image in images:
        feature_vector = LBP(image, 1, 8, mapping, 'h')
        feature_matrix.append(feature_vector)
    return np.array(feature_matrix)


def calculate_metrics(conf_mat):
    accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    precision = np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis=0))
    recall = np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score


def main():
    # Parameters
    lbp_type = 'riu2'
    desired_dimensions = 10
    train_ratio = 0.8

    # Load images
    image_folder = '../malimg_dataset'
    images, labels, categories = load_images(image_folder)

    # Create LBP mapping
    mapping = getmapping(8, lbp_type)

    # Extract LBP features
    feature_matrix = extract_lbp_features(images, mapping)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, labels, train_size=train_ratio, stratify=labels, random_state=42)

    # GMM Fitting
    num_categories = len(categories)
    gm_models = []
    valid_categories = []
    regularization_value = 0.01

    for category in categories:
        category_features = X_train[y_train == category]
        if len(category_features) < 2:
            print(f"Warning: Category '{
                  category}' has fewer than 2 samples and will be skipped.")
            continue

        try:
            gm = GaussianMixture(n_components=2, covariance_type='full',
                                 reg_covar=regularization_value, random_state=42)
            gm.fit(category_features[:, :10])
            gm_models.append(gm)
            valid_categories.append(category)
        except Exception as e:
            print(f"Error fitting GMM for category '{category}': {str(e)}")

    if len(gm_models) == 0:
        print("Error: No valid GMMs could be fitted. Check your data and parameters.")
        return

    # Update categories to only include valid ones
    categories = valid_categories
    num_categories = len(categories)

    # Create 2D GMMs for plotting (only for valid categories)
    gm_models_2d = []
    for category in categories:
        category_features = X_train[y_train == category]
        gm_2d = GaussianMixture(n_components=1, covariance_type='full',
                                reg_covar=regularization_value, random_state=42)
        gm_2d.fit(category_features[:, :2])
        gm_models_2d.append(gm_2d)

    # Plotting without contour lines
    plt.figure(figsize=(12, 8))
    for category, color in zip(categories, plt.cm.rainbow(np.linspace(0, 1, num_categories))):
        category_features = X_train[y_train == category]
        plt.scatter(category_features[:, 0], category_features[:, 1], c=[
                    color], label=category, alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot without Gaussian Mixture Model Contour Lines')
    plt.legend(loc='best')
    plt.show()

    # Plotting with contour lines
    plt.figure(figsize=(12, 8))
    for category, color, gm_2d in zip(categories, plt.cm.rainbow(np.linspace(0, 1, num_categories)), gm_models_2d):
        category_features = X_train[y_train == category]
        plt.scatter(category_features[:, 0], category_features[:, 1], c=[
                    color], label=category, alpha=0.7)

        x_min, x_max = category_features[:, 0].min(
        ) - 1, category_features[:, 0].max() + 1
        y_min, y_max = category_features[:, 1].min(
        ) - 1, category_features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = -gm_2d.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=5, colors=[
                    color], alpha=0.5, linewidths=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot with Gaussian Mixture Model Contour Lines')
    plt.legend(loc='best')
    plt.show()

    # Predicting Labels for Test Data
    predicted_labels = []
    for test_sample in X_test:
        log_likelihoods = [gm.score_samples(
            test_sample[:10].reshape(1, -1))[0] for gm in gm_models]
        predicted_category = categories[np.argmax(log_likelihoods)]
        predicted_labels.append(predicted_category)

    # Filter out test samples that belong to skipped categories
    valid_test_indices = [i for i, label in enumerate(
        y_test) if label in categories]
    y_test_valid = y_test[valid_test_indices]
    predicted_labels_valid = [predicted_labels[i] for i in valid_test_indices]

    # Evaluation Metrics
    conf_mat = confusion_matrix(y_test_valid, predicted_labels_valid)
    accuracy, precision, recall, f1 = calculate_metrics(conf_mat)

    print(f'Accuracy: {accuracy:.2%}')
    print(f'Precision: {precision:.2%}')
    print(f'Recall: {recall:.2%}')
    print(f'F1-Score: {f1:.2%}')

    # Confusion Matrix Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45, ha='right')
    plt.yticks(tick_marks, categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Visualizing Test Data
    plt.figure(figsize=(12, 8))
    for category, color in zip(categories, plt.cm.rainbow(np.linspace(0, 1, num_categories))):
        category_features = X_test[np.array(predicted_labels) == category]
        plt.scatter(category_features[:, 0], category_features[:, 1], c=[
                    color], label=category, alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of Test Data with Predicted Labels')
    plt.legend(loc='best')
    plt.show()

    # Calculate and plot accuracies for each GMM
    accuracies = []
    for i, category in enumerate(categories):
        category_samples = y_test_valid == category
        category_predictions = np.array(predicted_labels_valid)[
            category_samples]
        category_accuracy = accuracy_score(
            y_test_valid[category_samples], category_predictions)
        accuracies.append(category_accuracy)
        print(f'Accuracy for GMM {i+1} ({category}): {category_accuracy:.2%}')

    plt.figure(figsize=(12, 6))
    plt.bar(range(1, num_categories + 1), accuracies)
    plt.xlabel('GMM Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Each GMM on Test Data')
    plt.xticks(range(1, num_categories + 1),
               categories, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
