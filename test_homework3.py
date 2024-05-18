from pages import A_Explore_Preprocess_Data, B_Train_Model, C_Test_Model
import pandas as pd
import numpy as np
import string
import pytest
############## Assignment 3 Inputs #########
student_filepath = "./datasets/Amazon Product Reviews I.csv"
grader_filepath = "./test_files/Amazon Product Reviews I.csv"
student_df = pd.read_csv(student_filepath)




def contains_punctuation(s):
    if isinstance(s, str):
        return any(c in string.punctuation for c in s)
    else:
        return False


def assert_no_punctuation(df, columns):
    for col in columns:
        assert not df[col].apply(contains_punctuation).any()
# Checkpoint 1
@pytest.mark.checkpoint1
def test_remove_punctuation():
    student_df_copy = pd.read_csv(student_filepath)
    std_cl_df, _ = A_Explore_Preprocess_Data.clean_data(student_df_copy)
    std_rm_punc_df = A_Explore_Preprocess_Data.remove_punctuation(
        std_cl_df, ['reviews', 'title'])
    assert_no_punctuation(std_rm_punc_df, ['reviews', 'title'])


#checkpoint 3
@pytest.mark.checkpoint3
def test_tf_idf_encoder():
    student_df_copy = pd.read_csv(student_filepath)
    std_cl_df, _ = A_Explore_Preprocess_Data.clean_data(student_df_copy)
    std_rm_punc_df = A_Explore_Preprocess_Data.remove_punctuation(
        std_cl_df, ['reviews', 'title'])
    std_review_df, std_review_count_vect, std_review_transformer, std_review_tfidf_word_count_df = A_Explore_Preprocess_Data.tf_idf_encoder(
        std_rm_punc_df, 'reviews')

    expected_review_df = pd.read_pickle(
        "./test_files/tf_idf_encoder_review.pkl")
    pd.testing.assert_frame_equal(std_review_tfidf_word_count_df, expected_review_df)



# Checkpoint 2
@pytest.mark.checkpoint2
def test_word_count_encoder():
    student_df_copy = pd.read_csv(student_filepath)
    std_cl_df, _ = A_Explore_Preprocess_Data.clean_data(student_df_copy)
    std_rm_punc_df = A_Explore_Preprocess_Data.remove_punctuation(
        std_cl_df, ['reviews', 'title'])
    std_review_df, std_review_count_vect, std_review_word_count_df = A_Explore_Preprocess_Data.word_count_encoder(
        std_rm_punc_df, 'reviews', 'word', (1,1), stop_words=None)

    expected_review_df = pd.read_pickle(
        "./test_files/word_count_encoder_reviews.pkl")

    pd.testing.assert_frame_equal(std_review_word_count_df, expected_review_df)

# Checkpoint 4
@pytest.mark.checkpoint4
def test_split_dataset():
    df, _ = A_Explore_Preprocess_Data.clean_data(student_df)
    df = A_Explore_Preprocess_Data.remove_punctuation(df, ['reviews', 'title'])
    df, _, word_count_df = A_Explore_Preprocess_Data.word_count_encoder(df, 'reviews', 'word', (1,1), stop_words=None)
    df = B_Train_Model.set_pos_neg_reviews(df, 3)
    s_train_x, s_val_x, s_train_y, s_val_y = B_Train_Model.split_dataset(
        df, 1, 'sentiment', 'Word Count', random_state=42)
    
    expected_val_indices = [439, 111, 1047, 1176, 70, 352, 289, 374, 930, 167, 912]
    expected_train_indices = set(df.index.values) - set(expected_val_indices)

    assert list(s_val_x.index.values) == expected_val_indices
    assert list(s_val_y.index.values) == expected_val_indices
    assert set(s_train_x.index.values) == expected_train_indices
    assert set(s_train_y.index.values) == expected_train_indices


def preprocess_model_tests(df):
    df, _ = A_Explore_Preprocess_Data.clean_data(df)
    df = A_Explore_Preprocess_Data.remove_punctuation(df, ['reviews'])
    df, _, word_count_df = A_Explore_Preprocess_Data.word_count_encoder(df, 'reviews','word', (1,1), stop_words=None)
    
    df = pd.concat([df, word_count_df], axis=1)
    #df, _, word_count_df = A_Explore_Preprocess_Data.word_count_encoder(df, 'reviews', 'word', (1,1), stop_words=None)
    df = B_Train_Model.set_pos_neg_reviews(df, 3)
    X_train, X_val, y_train, y_val = B_Train_Model.split_dataset(
        df, 1, 'sentiment', 'Word Count')

    return X_train, X_val, y_train, y_val


# Checkpoint 5
@pytest.mark.checkpoint5
def test_logistic_regression_predict_probability():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    # turn dataframe to numpy array
    y_train = np.ravel(y_train)
    X_train = X_train.to_numpy()
    # params = {'max_iter': 1000,
    #           'penalty': 'l1', 'tol': 0.01, "solver": "liblinear"}
    student_lr = B_Train_Model.LogisticRegression(learning_rate=0.01, num_iterations=100)
    # student_lr = B_Train_Model.train_logistic_regression(
        # X_train, y_train, 'Logistic Regression', params)
    #student_lr.fit(X_train, y_train)
    student_lr.b = 0
    student_lr.num_features = X_train.shape[1]
    student_lr.W = np.zeros(student_lr.num_features)
    student_pred = student_lr.predict_probability(X_val)
    assert np.allclose(student_pred,
                    np.array([0.5, 0.5, 0.5 , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 , 0.5, 0.5]))
    # assert np.allclose(student_pred,
    #                    np.array([0.50549834, 0.50823491, 0.5021074 , 0.49984889, 0.50373106,
    #                     0.51010914, 0.50566721, 0.50121333, 0.5023748 , 0.50494358,
    #                     0.50108815]))
# Checkpoint 6
@pytest.mark.checkpoint6
def test_logistic_regression_compute_avg_log_likelihood():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    # turn dataframe to numpy array
    #y_train = y_train['sentiment'].to_numpy()
    y_train = np.ravel(y_train)
    X_train = X_train.to_numpy()
    # params = {'max_iter': 1000,
    #           'penalty': 'l1', 'tol': 0.01, "solver": "liblinear"}
    student_lr = B_Train_Model.LogisticRegression(learning_rate=0.01, num_iterations=100)
    # student_lr = B_Train_Model.train_logistic_regression(
        # X_train, y_train, 'Logistic Regression', params) compute_avg_log_likelihood(self.X, self.Y, self.W)
    #student_lr.fit(X_train, y_train)
    student_lr.num_features = X_train.shape[1]
    student_lr.W = np.zeros(student_lr.num_features)
    student_weights_df = student_lr.W
    student_lp = student_lr.compute_avg_log_likelihood(X_train, y_train, student_weights_df)

    expected_lp = -0.6931471805599452
    tol = 1e-6
    assert np.isclose(student_lp, expected_lp, rtol=tol)


# Checkpoint 7
@pytest.mark.checkpoint7
def test_logistic_regression_update_weights():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    student_lr = B_Train_Model.LogisticRegression(learning_rate=0.01, num_iterations=100)
    #y_train = y_train['sentiment'].to_numpy()
    y_train = np.ravel(y_train)
    X_train = X_train.to_numpy()
    student_lr.X = X_train
    student_lr.Y = y_train
    student_lr.b = 0
    student_lr.num_features = X_train.shape[1]
    student_lr.num_examples = X_train.shape[0]
    student_lr.W = np.zeros(student_lr.num_features)
    student_lr.update_weights()
    expected_weigts = np.load(
        "./test_files/Logistic_Regression_Update_Weight.npy")

    np.testing.assert_allclose(student_lr.W, expected_weigts, rtol=1e-5, atol=1e-8)

# Checkpoint 8
@pytest.mark.checkpoint8
def test_logistic_regression_predict():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    student_lr = B_Train_Model.LogisticRegression(learning_rate=0.01, num_iterations=100)
    #y_train = y_train['sentiment'].to_numpy()
    y_train = np.ravel(y_train)
    X_train = X_train.to_numpy()
    student_lr.X = X_train
    student_lr.Y = y_train
    student_lr.b = 0
    student_lr.num_features = X_train.shape[1]
    student_lr.num_examples = X_train.shape[0]
    student_lr.W = np.zeros(student_lr.num_features)
    student_y_pred = student_lr.predict(student_lr.X)
    assert (np.sum(student_y_pred)== -1042)
    
# Checkpoint 9
@pytest.mark.checkpoint9
def test_logistic_regression_fit():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    # turn dataframe to numpy array
    #y_train = y_train['sentiment'].to_numpy()
    y_train = np.ravel(y_train)
    X_train = X_train.to_numpy()
   
    

    student_lr = B_Train_Model.LogisticRegression(learning_rate=0.01, num_iterations=500)

    student_lr.fit(X_train, y_train)
    student_weights = student_lr.W
    expected_weights = np.load(
        "./test_files/Logistic_Regression_fit_Weight.npy")

    np.testing.assert_allclose(student_weights, expected_weights, rtol=1e-5, atol=1e-6)
# Checkpoint 10
@pytest.mark.checkpoint10
def test_logistic_regression_get_weight():
    student_lr = B_Train_Model.LogisticRegression(learning_rate=0.01, num_iterations=100)
    model_name = 'Logistic Regression'
    student_lr.W = np.array([1, -1, 0.5])
    out_dict = student_lr.get_weights(model_name=model_name)
    expected_dict = {'Logistic Regression': np.array([ 1. , -1. ,  0.5]), 'Stochastic Gradient Ascent with Logistic Regression': []}
    expected_keys = ['Logistic Regression', 'Stochastic Gradient Ascent with Logistic Regression']
    assert set(out_dict.keys()) == set(expected_keys), "keys not matching"
    assert np.array_equal(out_dict[model_name], expected_dict[model_name]),"Weights not matched"

# Checkpoint 11
np.random.seed(42)
@pytest.mark.checkpoint11
def test_StochasticLogisticRegression_fit():
    X_train, X_val, y_train, y_val = preprocess_model_tests(student_df)
    # turn dataframe to numpy array
    #y_train = y_train['sentiment'].to_numpy()
    y_train = np.ravel(y_train)
    X_train = X_train.to_numpy()

    student_Stlr = B_Train_Model.StochasticLogisticRegression(learning_rate=0.01, num_iterations=100, batch_size=50)

    student_Stlr.fit(X_train, y_train)
    student_weights = student_Stlr.W
    expected_weights = np.load(
        "./test_files/StochasticLogisticRegression_fit_Weight.npy")
    np.testing.assert_allclose(student_weights, expected_weights, rtol=1e-5, atol=1e-6)

# Checkpoint 12
@pytest.mark.checkpoint12
def test_compute_accuracy():
    prediction_labels = np.array([1,0,1,1,0,1,0,0])
    true_labels =       np.array([1,0,1,0,0,1,1,1])
    s_accuracy = C_Test_Model.compute_accuracy(prediction_labels,true_labels)

    assert s_accuracy == 0.625, "Incorrect Accuracy, Debug!! Look at the formula"



# Checkpoint 13
@pytest.mark.checkpoint13
def test_compute_precision_recall():
    prediction_labels = np.array([1,0,1,1,0,1,0,0])
    true_labels =       np.array([1,0,1,0,0,1,1,1])
    s_precision, s_recall = C_Test_Model.compute_precison_recall(prediction_labels, true_labels)
    assert s_precision == 0.75, "Incorrect Precision, Debug!! Look at the formula"
    assert s_recall == 0.6, "Incorrect recall, Debug!! Look at the formula"
