# Import the modules
from src.data.data_preparation import training_set, test_set, num_classes, class_names
from src.model.model_building import build_model
from src.model.model_training import train_model, plot_training_history
from src.model.model_evaluation import evaluate_model
from src.test.model_testing import test_model

def main():
    # Step 1: Building the model
    print("Building the model")
    model = build_model(input_shape=(124, 124, 3), num_classes=num_classes)
    print("Model built successfully")

    # Step 2: Training the model
    print("Starting model training")
    history = train_model(model, training_set, test_set, epochs=5)
    print("Model training completed")

    # Step 3: Plotting training history
    print("Plotting training history")
    plot_training_history(history)

    # Step 4: Evaluating the model
    print("Evaluating the model")
    evaluate_model(model, test_set, training_set)

    # Step 5: Testing the model on an image
    print("Testing the model with an image")
    test_model(model, 'data_original/real_test_set', class_names)

if __name__ == "__main__":
    main()
