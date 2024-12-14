def evaluate_model(model, test_set, training_set):
    test_result = model.evaluate(test_set, steps=1)
    train_result = model.evaluate(training_set, steps=1)

    print("Test-set classification accuracy: {0:.2%}".format(test_result[1]))
    print("Train-set classification accuracy: {0:.2%}".format(train_result[0]))
