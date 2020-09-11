from fairness.algorithms.Algorithm import Algorithm

class Generic(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        ## self.classifier should be set in any class that extends this one

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        # remove sensitive attributes from the training set
        train_df_nosensitive = train_df.drop(columns = sensitive_attrs)
        test_df_nosensitive = test_df.drop(columns = sensitive_attrs)

        # create and train the classifier
        y = train_df_nosensitive[class_attr]
        X = train_df_nosensitive.drop(columns = class_attr)
        
        classifier = self.get_classifier()        
        classifier.fit(X, y)

        # get the predictions on the test set
        X_test = test_df_nosensitive.drop(class_attr, axis=1)
        predictions = classifier.predict(X_test)

        # Get the predicted probabilities if classifier allows for this
        if self.predict_proba:
            prediction_probs = classifier.predict_proba(X_test)
            return predictions, [], prediction_probs

        return predictions, [], []

    def get_supported_data_types(self):
        return set(["numerical-binsensitive"])

    def get_classifier(self):
        """
        Returns the created SKLearn classifier object.
        """
        return self.classifier
