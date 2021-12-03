import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

import boto3, os

import streamlit as st
import base64

import logging
import sagemaker
from time import gmtime, strftime
from sagemaker.spark.processing import PySparkProcessor

import io
import re

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('Background-changed.png')

#Global variables
mybucket = '551grouproject6'


# Function for pre processing daa=ta locally
def preprocess_data(df,transformed_data_key):
    k = df[df['Rating'] > 10.0].index
    df = df.drop(k)
    print(k)
    print(df['Rating'].unique())
    df_raw = df.copy()

    i = df[df['Category'] == '1.9'].index
    df = df.drop(i)
    print(df['Rating'].value_counts().index.tolist())
    df["Rating"] = df["Rating"].fillna(df["Rating"].median())
    df['App'] = df['App'].apply(lambda x: re.sub('[^a-zA-Z0-9]', '', x))
    j = df[df['App'] == ''].index
    df = df.drop(j)
    df['Category'] = df['Category'].apply(lambda x: re.sub('[^a-zA-Z0-9]', '0', x))
    df['Genres'] = df['Genres'].apply(lambda x : re.sub('[^a-zA-Z0-9]','', x))
    # Data cleaning for "Size" column
    df['Size'] = df['Size'].map(lambda x: x.rstrip('M'))
    df['Size'] = df['Size'].map(lambda x: str(round((float(x.rstrip('k')) / 1024), 1)) if x[-1] == 'k' else x)
    df['Size'] = df['Size'].map(lambda x: np.nan if x.startswith('Varies') else x)
    # Data cleaning for "Installs" column
    df['Installs'] = df['Installs'].map(lambda x: x.rstrip('+'))
    df['Installs'] = df['Installs'].map(lambda x: ''.join(x.split(',')))
    # Data cleaning for "Price" column
    df['Price'] = df['Price'].map(lambda x: x.lstrip('$').rstrip())
    # Row [7312,8266] removed due to "Unrated" value in Content Rating
    i = df[df['Content Rating'] == 'Unrated'].index
    df.drop(df.index[i], inplace=True)

    df.drop(["Last Updated", "Current Ver", "Android Ver"], axis=1, inplace=True)

    df.dropna(axis=0, inplace=True)
    print(df.isnull().sum())
    df.to_csv(transformed_data_key, header=False, index=False)
    boto3.Session().resource('s3').Bucket(mybucket).Object(os.path.join(transformed_data_key)).upload_file(
        transformed_data_key)

    return df,df_raw


# Function to preprocess data using Spark
def spark_preprocess_data(key_for_data, file_data_key):
    sagemaker_logger = logging.getLogger("sagemaker")
    sagemaker_logger.setLevel(logging.INFO)
    sagemaker_logger.addHandler(logging.StreamHandler())

    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    role = "arn:aws:iam::660602874126:role/AmaazonSageMaker-ExecutionRole-1"

    # Upload the raw input dataset to a unique S3 location
    timestamp_prefix = strftime("%Y-%m", gmtime()) + key_for_data
    prefix = "sagemaker/spark-preprocess-demo/{}".format(timestamp_prefix)
    input_prefix_abalone = "{}/input/raw/googleplaystore_transformed".format(prefix)
    input_preprocessed_prefix_abalone = "{}/input/preprocessed/googleplaystore_transformed".format(prefix)
    file_data_key = file_data_key

    sagemaker_session.upload_data(
        path="googleplaystore_transformed.csv", bucket=bucket, key_prefix=input_prefix_abalone
    )

    # Run the processing job
    spark_processor = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version="2.4",
        role=role,
        instance_count=2,
        instance_type="ml.m5.xlarge",
        max_runtime_in_seconds=1200,
    )
    # print("./data/googleplaystore_transformed.csv", bucket,input_prefix_abalone, prefix)
    spark_processor.run(
       submit_app="./code/preprocess.py",
       arguments=[
           "--s3_input_bucket",
           bucket,
           "--s3_input_key_prefix",
           input_prefix_abalone,
           "--s3_output_bucket",
           bucket,
           "--s3_output_key_prefix",
           input_preprocessed_prefix_abalone,
           "--s3_output_file_key",
           file_data_key,
       ],
       spark_event_logs_s3_uri="s3://{}/{}/spark_event_logs".format(bucket, prefix)
    )

    return bucket,input_preprocessed_prefix_abalone


#Function to display the details and analyze the data using ML
def show_details_data_analysis(df, df_raw, bucket, input_preprocessed_prefix_abalone):
    st.subheader('Data Set')
    with st.expander("Raw Data"):
        st.write(df_raw.head(12))

    st.sidebar.subheader('Data Features')
    buffer = io.StringIO()
    df_raw.info(buf=buffer)
    s = buffer.getvalue()
    st.sidebar.text(s)
    st.sidebar.text(df.describe())

    preprocess_data_location = 's3://{}/{}/train/part-00000'.format(bucket, input_preprocessed_prefix_abalone)
    df_preproc = pd.read_csv(preprocess_data_location, header=None)
    df_preproc.columns = ['Rating', 'App', 'Category', 'Type', 'Content Rating', 'Genres', 'Reviews', 'Size',
                          'Installs', 'Price', 'App_Label', 'Category_Label', 'Genres_Label']
    # Category features encoding
    # df_preproc = df_preproc_with_label[['Rating', 'App', 'Category', 'Type', 'Content Rating', 'Genres', 'Reviews', 'Size',
    #                       'Installs', 'Price']]
    category_list = df_preproc['Category'].unique().tolist()
    category_list = ['cat_' + str(word) for word in category_list]
    df_preproc = pd.concat([df_preproc, pd.get_dummies(df_preproc['Category'], prefix='cat')], axis=1)

    top_genres = df_raw.Genres.value_counts().reset_index().rename(columns={'Genres': 'Count', 'index': 'Genres'})
    genres_installs = df_raw.groupby(['Genres'])[['Installs']].sum()
    top_genres_installs = pd.merge(top_genres, genres_installs, on='Genres')
    top_20_genres_installs = top_genres_installs.head(20)
    genres_ratings_df = df_raw.groupby(['Genres'])[['Rating']].mean()
    print(genres_ratings_df.sort_values(by='Rating', ascending=False))
    genres_installs_ratings = pd.merge(top_genres_installs, genres_ratings_df, on='Genres')
    print(genres_installs_ratings['Rating'].describe())
    print(genres_installs_ratings['Rating'].value_counts().index.tolist())

    with st.expander("Preprocessed Data"):
        st.write(df_preproc.head(12))

    with st.expander("Scatter plot of each features based on Type in the dataset"):
        fig = sns.pairplot(df_preproc[['Rating', 'Size', 'Installs', 'Reviews', 'Price', 'Type']], hue="Type")
        st.pyplot(fig)

    with st.expander("Top 20 Genres Plot"):
        barplot20 = plt.figure(figsize=(14, 7))
        plt.xticks(rotation=65)
        plt.xlabel("Genres")
        plt.ylabel("Number of application")
        sns.barplot(top_20_genres_installs.Genres, top_20_genres_installs.Count)
        st.pyplot(barplot20)

    with st.expander('Distribution of Rating'):
        g = plt.figure(figsize=(14, 7))
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        sns.kdeplot(genres_installs_ratings.Rating, color="Red", shade=True)
        st.pyplot(g)

    with st.expander("Heat Map"):
        heatmap = sns.heatmap(df_preproc.corr(), vmin=-1, vmax=1)
        data_corr = df_preproc[['Rating', 'Size', 'Installs', 'Reviews', 'Price', 'Type', 'Category', 'Genres']]
        heatmap = plt.figure(figsize=(20, 15))
        heatmap, ax = plt.subplots()
        sns.heatmap(data_corr.corr(), vmin=-1, vmax=1)
        st.write(heatmap)

    df_X = df_preproc[df_preproc.columns[1:]]
    df_y = df_preproc[df_preproc.columns[0]]
    # df_X.drop(["App", "Content Rating", "Category"], axis=1, inplace=True)
    # X_with_intercept = sm.add_constant(df_X)
    # print(df_X)
    # st.write(df_X)
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
    # print(X_test)
    X_test_raw = X_test[['App_Label', 'Category_Label', 'Genres_Label', 'Type', 'Reviews', 'Size', 'Installs', 'Price']]
    X_train.drop(["App", "Content Rating", "Category", 'App_Label', 'Category_Label', 'Genres_Label'], axis=1, inplace=True)
    X_test.drop(["App", "Content Rating", "Category", 'App_Label', 'Category_Label', 'Genres_Label'], axis=1, inplace=True)
    # print(df_preproc['Category'].dtypes)
    # print(type(df_X))
    # print(type(X_train['Category'].dtypes))

    rf = RandomForestClassifier()
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [20, 30, 40, 50], "max_features": ['sqrt']}

    # df_X.shape[0] == df_y.shape[0]
    # grid_search = GridSearchCV(rf, param_grid=param_grid)
    # grid_search.fit(X_train, y_train)

    # st.write("Best score is {}".format(grid_search.best_params_))
    # st.write("Best score is {}".format(grid_search.best_score_))

    # Randomforest CLassifier Train
    # rf = RandomForestClassifier(max_depth = 20, n_estimators = 100)
    # rf.fit(X_train,y_train)
    # y_pred = rf.predict(X_train)

    # acc = accuracy_score(y_train, y_pred)
    # st.write("Random Forest Train data accuracy: {:.2f}".format(acc))

    # RandomForest Classifier Test
    # rf.fit(X_train,y_train)
    # y_pred = rf.predict(X_test)

    # acc = accuracy_score(y_test, y_pred)
    # st.write("Random Forest Test data accuracy: {:.5f}".format(acc))

    # std=StandardScaler()
    # X_train_std = std.fit_transform(X_train)
    # X_test_std = std.transform(X_test)
    # y_train = list(y_train)
    # y_test = list(y_test)
    # model = LinearRegression()
    # model = KNeighborsRegressor(n_neighbors=15)
    # model.fit(X_train, y_train)
    # print(results.summary())
    # accuracy = model.score(X_test,y_test)
    # st.write('Accuracy: ' + str(np.round(accuracy*100, 2)) + '%')

    alpha = [10 ** i for i in range(-5, 5)]
    l1_clf = GridSearchCV(estimator=xgb.XGBRegressor(n_estimators=100, max_depth=4)
                          , param_grid={'alpha': alpha}
                          , cv=KFold(10, shuffle=True, random_state=155))

    l1_clf.fit(X_train, y_train)
    best_alpha_l1 = l1_clf.best_params_['alpha']
    l1_clf_best = xgb.XGBRegressor(n_estimators=1000, max_depth=1, reg_alpha=best_alpha_l1, learning_rate=0.01,
                                   subsample=0.8, gamma=0, colsample_bytree=0.7).fit(X_train, y_train)
    y_pred = l1_clf_best.predict(X_test)
    # rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # mse = (metrics.mean_squared_error(y_test, y_pred))
    # st.write("Root Mean Square error is %f" % (rmse))
    # st.write("Mean Square error is ", mse)
    # st.write("R^2 score is ", r2_score(y_test, y_pred))
    from math import ceil
    xgbr = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            colsample_bynode=1, colsample_bytree=1, gamma=0,
                            importance_type='gain', learning_rate=0.1, max_delta_step=0,
                            max_depth=3, min_child_weight=1, missing=1, n_estimators=100,
                            n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                            silent=None, subsample=1, verbosity=1)
    xgbr.fit(X_train, y_train)
    # score = xgbr.score(X_train, y_train)
    # st.write("Training score: ", score)
    # scores = cross_val_score(xgbr, X_train, y_train, cv=10)
    # st.write("Mean cross-validation score: %.2f" % scores.mean())
    ypred = xgbr.predict(X_test)
    df_final = X_test_raw.copy()
    # df_final['Y-true'] = y_test
    df_final['Y-pred'] = ypred
    df_final['Y-pred'] = df_final['Y-pred'].apply(lambda x : ceil(x * 10) / 10.0)
    df_final['Category_Label'] = df_final['Category_Label'].apply(lambda x: x.replace('0','-'))
    # df_final = pd.concat([X_test,y_test, ypred],axis=1)
    df_final = df_final.sort_values(by='Y-pred', ascending=False)
    with st.expander("Predicted Data Set"):
        st.write(df_final)

    # with st.expander("Predicted Values Plot"):
    #     # scatter = plt.figure(figsize=(14, 7))
    #     df_final['Y-true'] = y_test
    #     # sns.scatterplot(df_final['Y-pred'], df_final['Y-true'])
    #     ax = df_final.plot(kind='scatter', x=[0,1,2,3,4,5], y='Y-pred', color='r')
    #     df_final.plot(kind='scatter', x=[0,1,2,3,4,5], y='Y-true', color='g', ax=ax)
    #     # plt.scatter(x=Y, df_final['Y-pred'], c='b', marker="s")
    #     # plt.scatter([0,1,2,3,4,5], df_final['Y-true'], c='r', marker="o")
    #     plt.legend(['Predicted Values', 'True Values'])
    #     st.pyplot(ax)
    with st.expander("Top Recommended apps based on Category"):
        g = df_final.groupby(["Category_Label"]).apply(
            lambda x: x.sort_values(["Y-pred"], ascending=False)).reset_index(drop=True)
        # select top N rows within each continent
        st.write(g.groupby('Category_Label').head(5))
    # print(ypred)
    # print(df_final['Y-pred'])
    # mse = metrics.mean_squared_error(y_test, ypred)
    # st.write("MSE: %.2f" % mse)
    # st.write("RMSE: %.2f" % (mse ** (1 / 2.0)))

    # st.write("Results")
    # st.write(results.summary())
    # y_pred_test = model.predict(X_test)
    # st.write("Mean Squared Error for test")
    # st.write(metrics.mean_squared_error(y_test,y_pred_test))

    # n_neighbors = np.arange(1, 50, 1)
    # scores = []
    # for n in n_neighbors:
    #    model.set_params(n_neighbors=n)
    #    model.fit(X_train, y_train)
    #    scores.append(model.score(X_test, y_test))

    # fig1 = plt.figure(figsize=(7, 5))
    # plt.title("Effect of Estimators")
    # plt.xlabel("Number of Neighbors K")
    # plt.ylabel("Score")
    # plt.plot(n_neighbors, scores)
    # st.pyplot(fig1)
    # y_pred_train = model.predict(X_train)
    # st.write("Mean Squared Error for train")
    # st.write(metrics.mean_squared_error(y_train,y_pred_train))

    # results = model.fit(X_train,y_train)
    # y_pred = results.predict(X_test)
    # st.write("Test error for fitting linear model using least squares:",metrics.mean_squared_error(y_test,y_pred))
    # st.write("R^2 score: ",metrics.r2_score(y_test, df_final['Y-pred']))


# Create a page dropdown
page = st.selectbox("Choose your page", ["CSV Data Analysis - Textual Data", "Sentiment Analysis - Non Textual Data - PDF"])
if page == "CSV Data Analysis - Textual Data":
    # Display details of page 1
    st.write("""

    # Analysis of Google Play Store Apps using Machine Learning

    Analysis of Google Play Store Apps data is presented below using Machine Learning algorithms of Regression. The data set used can be found
    on Kaggle at https://www.kaggle.com/lava18/google-play-store-apps. Information about the dataset can be seen in the SideBar.

    Additionally the user can upload a dataset in the csv from the SideBar using "Browse File" option. The user can view 
    visualizations and inferences by expanding each titles. User can also view the top recommended apps by selecting/expanding
    the option below.

    """)

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        # file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.sidebar.text("File Name:")
        st.sidebar.text(uploaded_file.name)
        st.sidebar.text("File Type:")
        st.sidebar.text(uploaded_file.type)
        st.sidebar.text("File Size:")
        st.sidebar.text(uploaded_file.size)
        input_df = pd.read_csv(uploaded_file)
        data_key = uploaded_file.name + '.csv'
        st.sidebar.text(data_key)
        transformed_data_key = uploaded_file.name + '_transformed.csv'
        st.sidebar.text(transformed_data_key)
        df,df_raw = preprocess_data(input_df,transformed_data_key)
        bucket,input_preprocessed_prefix_abalone = spark_preprocess_data("custom data",transformed_data_key)
        show_details_data_analysis(df, df_raw, bucket, input_preprocessed_prefix_abalone)

    else:
        data_key = 'googleplaystore.csv'
        transformed_data_key = 'googleplaystore_transformed.csv'
        data_location = 's3://{}/{}'.format(mybucket, data_key)
        input_df = pd.read_csv(data_location)
        df,df_raw = preprocess_data(input_df, transformed_data_key)
        bucket,input_preprocessed_prefix_abalone = spark_preprocess_data("default data",transformed_data_key)
        show_details_data_analysis(df, df_raw, bucket, input_preprocessed_prefix_abalone)

# Sentiment Analysis
elif page == "Sentiment Analysis - Non Textual Data - PDF":
    # Display details of page 2
    st.write("""

        # Analysis of Google Play Store App Review using Sentiment Analysis

        Analysis of Google Play Store Apps Review from a PDF File is presented below using Sentiment Analysis. 
        The data is manually created. Information about the cleaned and processed data can be seen in the SideBar.

        The user can view the analysis and visualization of the data below by expanding each titles.

        """)
    import PyPDF2
    import pandas as pd
    from textblob import TextBlob
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    # creating an object
    file = open('rawdata4.pdf', 'rb')

    # creating a pdf reader object
    fileReader = PyPDF2.PdfFileReader(file)


    def get_Sentiment(text):
        blob = TextBlob(text)
        sentimentPolarity = blob.sentiment.polarity
        sentimentSubjectivity = blob.sentiment.subjectivity
        sentiment = [sentimentPolarity, sentimentSubjectivity]
        return sentiment


    fulltext = ''
    for page in range(fileReader.numPages):
        pageObj = fileReader.getPage(page)
        pagetxt = pageObj.extractText()
        fulltext += pagetxt
    fulltext = fulltext.replace('\n', '')

    # maindf = pd.DataFrame()
    blob = []
    count = 0
    dftext = fulltext.split('#')
    for line in range(len(dftext)):
        fl = []
        l = dftext[line].split('@')
        l = [x.strip() for x in l]
        if len(l) != 2:
            continue
        polarity, subjectivity = get_Sentiment(l[-1])
        l.append(polarity)
        l.append(subjectivity)
        blob.append(l)
    df_pdf = pd.DataFrame(blob)

    df_pdf.columns = ['App', 'Review', 'Polarity', 'Subjectivity']

    removelist = df_pdf['App'].value_counts().index.tolist()[23:]
    idx = []
    for x in removelist:
        idx.extend(df_pdf[df_pdf['App'] == x].index)
    for i in idx:
        df_pdf = df_pdf.drop(i)

    # In[7]:

    df_pdf['Sentiment'] = df_pdf['Polarity']
    df_pdf.loc[df_pdf['Polarity'] == 0.0, 'Sentiment'] = 'Neutral'
    df_pdf.loc[df_pdf['Polarity'] < 0.0, 'Sentiment'] = 'Negative'
    df_pdf.loc[df_pdf['Polarity'] > 0.0, 'Sentiment'] = 'Positive'

    st.subheader("Non Textual Data: PDF")
    with st.expander("Cleaned data from the file"):
        st.write(df_pdf.head(10))

    st.sidebar.subheader('Metadata of Cleaned Dataset')
    import io

    buffer = io.StringIO()
    df_pdf.info(buf=buffer)
    s = buffer.getvalue()
    st.sidebar.text(s)
    st.sidebar.text(df_pdf.describe())

    pos = df_pdf.groupby('App')['Polarity'].apply(lambda x: (x >= 0.0).sum()).reset_index(name='count')
    neg = df_pdf.groupby('App')['Polarity'].apply(lambda x: (x < 0.0).sum()).reset_index(name='count')

    N = len(pos)
    positive = pos['count']
    negative = neg['count']
    labels = ['-'.join(x.split('.')[-2:]) for x in pos['App'].tolist()]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4

    with st.expander("Distribution of Positive & Negative reviews per App"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(ind, positive, width, color='b')
        ax.bar(ind, negative, width, bottom=positive, color='r')
        ax.set_ylabel('Reviews')
        ax.set_xlabel('Apps')
        plt.xticks(ind, labels)
        plt.xticks(rotation=90)
        ax.set_yticks(np.arange(0, 81, 10))
        ax.legend(labels=['Positive', 'Negative'])
        st.pyplot(fig)

    with st.expander("Distribution of Subjectivity"):
        histsub = plt.figure(figsize=(18, 9))
        plt.xlabel("Subjectivity")
        plt.hist(df_pdf[df_pdf['Subjectivity'].notnull()]['Subjectivity'])
        st.pyplot(histsub)

    # It can be seen that maximum number of sentiment subjectivity lies between 0.4 to 0.6. From this we can conclude that maximum number of users give reviews to the applications, according to their experience.

    merged_df = df_pdf.copy()
    merged_df['Subjectivity'] = merged_df['Subjectivity'].abs()
    merged_df['Polarity'] = merged_df['Polarity'].abs()

    with st.expander("Is sentiment_subjectivity proportional to sentiment_polarity"):
        scatter = plt.figure(figsize=(14, 7))
        sns.scatterplot(merged_df['Subjectivity'], merged_df['Polarity'])
        st.pyplot(scatter)
    # From the above scatter plot it can be concluded that sentiment subjectivity is not always proportional to sentiment polarity but in maximum number of case, shows a proportional behavior, when variance is too high or low

    with st.expander("A Pie Chart Representing Percentage of Review Sentiments"):
        counts = list(df_pdf['Sentiment'].value_counts())
        labels = 'Positive Reviews', 'Negetive Reviews', 'Neutral Reviews'
        matplotlib.rcParams['font.size'] = 12
        matplotlib.rcParams['figure.figsize'] = (8, 8)
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=labels, explode=[0, 0.05, 0.005], shadow=True, autopct="%.2f%%")
        ax1.axis('off')
        ax1.legend()
        st.pyplot(fig1)
