import pandas as pd                   
import streamlit as st                 
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from helper_functions import fetch_dataset, clean_data, display_review_keyword

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################

# Checkpoint 1
def remove_punctuation(df, features):
    """
    This function removes punctuation from features (i.e., product reviews)

    Input: 
        - df: the pandas dataframe
        - feature: the features to remove punctation
    Output: 
        - df: dataframe with updated feature with removed punctuation
    """
    translator = str.maketrans('', '', string.punctuation)
    for feature_name in features:
        # check if the feature contains string or not
        if df[feature_name].dtype == 'object':
            # applying translate method eliminating punctuations
            df[feature_name] = df[feature_name].apply(
                lambda x: x.translate(translator) if isinstance(x, str) else x)

    # Confirmation statement
    st.write('Punctuation was removed from {}'.format(features))
    return df

# Checkpoint 2
def word_count_encoder(df, feature, analyzer='word', ngram_range=(1, 1), stop_words=None):
    """
    This function performs word count encoding on feature in the dataframe

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform word count encoding
        - analyzer: 'word', 'char', 'char_wb'
        - ngram_range (tuple): unigram - (1,1), unigram & bigram - (1,2), bigram - (2,2)
        - stop_words: stop_words (list) or 'english' (string)
    Output: 
        - df: dataframe with word count feature
        - count_vect: CountVectorizer
        - word_count_df: pandas dataframe with prefix 'word_count_'
    """
    # Create CountVectorizer object using analyzer, ngram_range, and stop_words
    count_vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)
    X_train_counts = count_vect.fit_transform(df[feature])
    word_count_df = pd.DataFrame(X_train_counts.toarray())
    word_count_df = word_count_df.add_prefix('word_count_')
    
    # Show confirmation statement
    st.write('Feature {} has been word count encoded from {} reviews.'.format(
        feature, len(word_count_df)))

    # Store new features in st.session_state
    st.session_state['data'] = df

    return df, count_vect, word_count_df

# Checkpoint 3
def tf_idf_encoder(df, feature, analyzer='word', ngram_range=(1, 1), stop_words=None, norm=None):
    """
    This function performs tf-idf encoding on the given features

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform tf-idf encoding
        - analyzer: 'word', 'char', 'char_wb'
        - ngram_range (tuple): unigram - (1,1), unigram & bigram - (1,2), bigram - (2,2)
        - stop_words: stop_words (list) or 'english' (string)
        - norm: 'l2': Sum of squares, 'l1': Sum of absolute values, 'n' for no normalization
    Output: 
        - df: dataframe with tf-idf encoded feature
        - count_vect: CountVectorizer
        - tfidf_transformer: TfidfTransformer with normalization in norm
        - tfidf_df: pandas dataframe with prefix 'tf_idf_word_count_'
    """
    # Create CountVectorizer object using analyzer, ngram_range, and stop_words
    count_vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)

    # Create TfidfTransformer object
    tfidf_transformer = TfidfTransformer(norm=norm)

    X_train_counts = count_vect.fit_transform(df[feature])
    #tfidf_feature_names = np.array(count_vect.get_feature_names_out())
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
    tfidf_df = tfidf_df.add_prefix('tf_idf_word_count_')
    
    # Show confirmation statement
    st.write(
        'Feature {} has been TF-IDF encoded from {} reviews.'.format(feature, len(tfidf_df)))

    # Store new features in st.session_state
    st.session_state['data'] = df

    return df, count_vect, tfidf_transformer, tfidf_df

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('You have uploaded the Amazon Product Reviews dataset. Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. See the unprocesses dataset below.')

    st.dataframe(df)

    # Remove irrelevant features
    df, data_cleaned = clean_data(df)
    if (data_cleaned):
        st.markdown('The dataset has been cleaned. Your welcome!')

    ############## Task 1: Remove Punctation
    st.markdown('### Remove punctuation from features')
    removed_p_features = st.multiselect(
        'Select features to remove punctuation',
        df.columns,
    )
    if (removed_p_features):
        df = remove_punctuation(df, removed_p_features)
        # Store new features in st.session_state
        st.session_state['data'] = df
        # Display updated dataframe
        st.dataframe(df)
        st.write('Punctuation was removed from {}'.format(removed_p_features))

    # Use stopwords or 'engligh'
    st.markdown('### Use stop words or not')
    use_stop_words = st.multiselect(
        'Use stop words?',
        ['Use stop words', 'english'],
    )
    st.write('You selected {}'.format(use_stop_words))
    
    stop_words_list=[]
    st.session_state['stopwords']='english'
    if('Use stop words' in use_stop_words):
        stopwords_file = st.file_uploader(
            'Upload stop words file', type=['csv', 'txt'])
        # Read file
        if(stopwords_file):
            stop_words_df = pd.read_csv(stopwords_file)
            #stop_words = list(np.array(stop_words_df.to_numpy()).reshape(-1))
            # Save stop words to session_state
            st.session_state['stopwords'] = list(np.array(stop_words_df.to_numpy()).reshape(-1))
            st.write('Stop words saved to session_state.')
            st.table(stop_words_df.head())

    if('english' in use_stop_words):
        st.session_state['stopwords'] = 'english'
        st.write('No stop words saved to session_state.')

    # Inspect Reviews
    st.markdown('### Inspect Reviews')

    review_keyword = st.text_input(
        "Enter a keyword to search in reviews",
        key="review_keyword",
    )

    # Display dataset
    st.dataframe(df)

    if (review_keyword):
        displaying_review = display_review_keyword(df, review_keyword)
        st.write('Summary of search results:')
        st.write('Number of reviews: {}'.format(len(displaying_review)))
        st.write(displaying_review)

    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)
    word_encoder = []

    # Initialize word encoders in session state
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = {'word_count':None, 'tfidf':None}
    st.session_state['tfidf_transformer'] = None

    word_count_col, tf_idf_col = st.columns(2)

    wc_analyzer = 'word'
    wc_n_ngram = (1,1)
    n_gram = {'unigram': (1,1), 'bigram': (2,2), 'unigram-bigram': (1,2)}
    ############## Task 2: Perform Word Count Encoding
    with (word_count_col):
        text_feature_select_int = st.selectbox(
            'Select text features for encoding word count',
            string_columns,
        )
        st.write('You selected feature: {}'.format(text_feature_select_int))

        wc_analyzer = st.selectbox(
            'Select the analyzer for encoding word count',
            ['word', 'char', 'char_wb'],
        )
        st.write('You selected analyzer: {}'.format(wc_analyzer))

        wc_n_ngram = st.selectbox(
            'Select n-gram for encoding word count',
            ['unigram', 'bigram', 'unigram-bigram'],
        )
        st.write('You selected n-gram: {}'.format(wc_n_ngram))

        if (text_feature_select_int and st.button('Word Count Encoder')):
            df, wc_count_vect, word_count_df = word_count_encoder(df, text_feature_select_int, analyzer=wc_analyzer, ngram_range=n_gram[wc_n_ngram], stop_words=st.session_state['stopwords'])
            word_encoder.append('Word Count')
            st.session_state['word_encoder'] = word_encoder
            st.session_state['count_vect']['word_count'] = wc_count_vect
            st.session_state['word_count_df']=word_count_df

            if('tfidf_word_count_df' in st.session_state):
                tfidf_word_count_df = st.session_state['tfidf_word_count_df']
                df = pd.concat([df, word_count_df, tfidf_word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df
            else:
                df = pd.concat([df, word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df

    ############## Task 3: Perform TF-IDF Encoding
    with (tf_idf_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for encoding TF-IDF',
            string_columns,
        )
        st.write('You selected feature: {}'.format(text_feature_select_onehot))

        tfidf_analyzer = st.selectbox(
            'Select the analyzer for encoding tfidf count',
            ['word', 'char', 'char_wb'],
        )
        st.write('You selected analyzer: {}'.format(tfidf_analyzer))

        tfidf_n_ngram = st.selectbox(
            'Select n-gram for encoding tfidf count',
            ['unigram', 'bigram', 'unigram-bigram'],
        )
        st.write('You selected n-gram: {}'.format(tfidf_n_ngram))

        if (text_feature_select_onehot and st.button('TF-IDF Encoder')):
            df, tfidf_count_vect, tfidf_transformer, tfidf_word_count_df = tf_idf_encoder(df, text_feature_select_onehot, analyzer=tfidf_analyzer, ngram_range=n_gram[tfidf_n_ngram], stop_words=st.session_state['stopwords'])
            word_encoder.append('TF-IDF')
            st.session_state['word_encoder'] = word_encoder
            st.session_state['count_vect']['tfidf'] = tfidf_count_vect
            st.session_state['transformer'] = tfidf_transformer
            st.session_state['tfidf_word_count_df']=tfidf_word_count_df

            if('word_count_df' in st.session_state):
                word_count_df = st.session_state['word_count_df']
                df = pd.concat([df, word_count_df, tfidf_word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df
            else:
                df = pd.concat([df, word_count_df], axis=1)
                # Store new features in st.session_state
                st.session_state['data'] = df

    # Display dataset
    st.dataframe(df)
    
    # Save dataset in session_state
    st.session_state['data'] = df

    st.write('Continue to Train Model')