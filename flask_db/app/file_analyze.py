import string
import re
from bs4 import BeautifulSoup
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
import io
import base64
import seaborn as sn
import requests
# from your_flask_app import db, Review

import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from textblob import TextBlob
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')


def airline_func(airline_name, selectSentiment):      # Web-scraping begin
    formatted_name = '-'.join(airline_name.split())
    base_url = f"https://www.airlinequality.com/airline-reviews/{formatted_name}"    # output_name // link address for web scraping
    pages = 10
    page_size = 100

    reviews = []

    for i in range(1, pages + 1):
        # print(f"Scraping page {i}")
        url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"
        response = requests.get(url)

        # Parse content
        content = response.content
        parsed_content = BeautifulSoup(content, 'html.parser')
        for para in parsed_content.find_all("div", {"class": "text_content"}):
            reviews.append(para.get_text())
        
    #=============     web-scraping end     =============== 

    # ==============   data cleaning start   ====================
    df = pd.DataFrame()
    df["reviews"] = reviews

    # set index
    df.index = range(1, len(df)+1)

    unnecessary_statement1 = '✅ Trip Verified | '
    unnecessary_statement2 = 'Not Verified | '
    unnecessary_word3 = '✅ Verified Review | '

    df.reviews = df.reviews.apply(lambda x: x.replace(unnecessary_statement1, ''))
    df.reviews = df.reviews.apply(lambda x: x.replace(unnecessary_statement2, ''))
    df.reviews = df.reviews.apply(lambda x: x.replace(unnecessary_word3, ''))

    # ==========        data cleaning stop    =================== 


    # randcol = random.choice(df['reviews'])   #  untuk mencari RANDOM REVIEWS
    
    # return randcol   # print out RANDOM REVIEWS 


    # ==============             DATA ANALYSIS       =================================  diagram deyatelnosty atau diagram posledovatelnos
    # buat untuk step by step
    
    # ==============        data analyze/analyzing start      =====================
    def analyze_sentiment(reviews):
        tokens = nltk.word_tokenize(reviews)
            
        tagged_tokens = nltk.pos_tag(tokens)

        lemmatized_words = []    
        lemmatizer = nltk.WordNetLemmatizer()
        for word, tag in tagged_tokens:
            tag = tag[0].lower() if tag[0].lower() in ['a', 'r', 'n', 'v'] else 'n'
            lemmatized_words.append(lemmatizer.lemmatize(word, tag))    

        cust_stopwords = nltk.corpus.stopwords.words('english')
        clean_words = [word for word in lemmatized_words if word.lower() not in cust_stopwords]

        clean_text = ' '.join(clean_words)

        blob = TextBlob(clean_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, polarity, subjectivity

    df[['sentiment', 'polarity', 'subjectivity']] = df['reviews'].apply(analyze_sentiment).apply(pd.Series)

    # ==============        data analyze/analyzing stop      =====================

    # print('Please select which sentiment reviews you want to display')
    # review_option = input('positive / negative / neutral?\n')



    # ==============        analyzing sentiment choice start      =====================

    def func_selectSentiment(selectSentiment):
        lis_res_sentiment = []
        positive_value = ['Positive', 'positive']
        negative_value = ['Negative', 'negative']

        if selectSentiment in positive_value:
            positive_reviews = df[df['sentiment'] == 'Positive']
            result_positive = np.random.choice(positive_reviews['reviews'])
            lis_res_sentiment.append(result_positive)
        elif selectSentiment in negative_value:   # negative_review = 'Negative'        #===============================
            negative_reviews = df[df['sentiment'] == 'Negative']
            result_negative = np.random.choice(negative_reviews['reviews'])
            lis_res_sentiment.append(result_negative)

        return lis_res_sentiment[0]
    
    res_sentiment = func_selectSentiment(selectSentiment=selectSentiment)
    # res_sentiment = "Positive" if selectSentiment == "positive" else "Negative"

    # ==============        analyzing sentiment choice stop      =====================

    plots =[]


#  ==================    POSITIVE    ==================  ===
# print('Please select which sentiment reviews you want to analyze')
    # review_option = 'Positive'



    # ==============        visualizing data start      =====================

    sentiment_reviews = df[df['sentiment'] == 'Positive']
    sentiment_reviews = sentiment_reviews['reviews'].tolist()
    sentiment_reviews = ' '.join(sentiment_reviews)
    text = sentiment_reviews
    # text
    text = text.lower()
    text = re.sub(r'\d+','', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = [word for word in text.split()]
    # nltk.download()
    custom_stopwords = stopwords.words('english')
    clean_tokens = tokens[:]

    for token in tokens:
        if token in custom_stopwords:   #words('english'):
            clean_tokens.remove(token)

    # freq = nltk.FreqDist(clean_tokens)
    freq_dist = FreqDist(clean_tokens)
    # for key, val in freq.items():
    #     print(str(key), ':', str(val))

    # # # ==============   VISUALISASI DATA     =============
    # # # =================================================================
    # # freq_pict = freq.plot(30)       
    # # freq.plot(30, )   #cumulative=False


    most_common_words = freq_dist.most_common(20)  #show

    words , frequencies = zip(*most_common_words)

    # plt.figure(figsize=(10,6))
    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency PLot')
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_png = buffer.getvalue()
    plot = base64.b64encode(image_png).decode()
    plots.append(plot)
    buffer.close()
    
    plt.clf()      #  ========================================================================

    # lis = []
    # freq = nltk.FreqDist(clean_tokens)
    # for key in freq.items():
    #     lis.append(str(key[0]))

    # print(lis)
    # print('==================\n')


    # def visual_pict(lis):
    #     stopwords= set(STOPWORDS)
    #     wordcloud = WordCloud(width=800, height=800,
    #                         background_color='white', stopwords=stopwords,
    #                         max_words=50,
    #                         min_font_size=10).generate(' '.join(lis))

    #     plt.figure(figsize= (8,8), facecolor=None)
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis('off')
    #     plt.tight_layout(pad=0)
    #     show_pict = plt.show()

    #     return show_pict
    
    # res_fin = visual_pict(lis)

    # datasets=[
    #     {'type':'bar', 'data': df.groupby('sentiment')['reviews'].count()},
    #     {'type': 'pie', 'data': df.groupby(['sentiment']).sum()['subjectivity']}
    # ]

    # df = pd.DataFrame({
    #     'sentiment': ['Positive', "Neutral", 'Negative'],
    #     'subjectivity':[100,200,150],
    #     'reviews': [300,500,400]
    # })
    
    # BAR CHART
    # tipe_subjectivity = df.groupby(['sentiment']).sum()['subjectivity']
    # tipe_subjectivity.plot(kind='bar')
    # plt.ylabel('Subjectivity')
    # plt.xlabel('Sentiment')
    # plt.title('Sentiment across Subjectivity')

    bar_sentiment_counts = df.groupby('sentiment')['reviews'].count()
    sn.barplot(x=bar_sentiment_counts.index, y=bar_sentiment_counts.values)
    plt.title('Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')

    # print('Please select which sentiment reviews you want to analyze')
    # review_option = input('Positive / Negative / Neutral?\n')

    # sentiment_reviews = df[df['sentiment'] == positive_option]
    # sentiment_reviews = sentiment_reviews['reviews'].tolist()
    # sentiment_reviews = ' '.join(sentiment_reviews)
    # text = sentiment_reviews
    # text = text.lower()
    # text = re.sub(r'\d+','', text)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    # tokens = [word for word in text.split()]

    # # nltk.download()
    # clean_tokens = tokens[:]
    # for token in tokens:
    #     if token in stopwords.words('english'):
    #         clean_tokens.remove(token)

    # freq = nltk.FreqDist(clean_tokens)
    # for key, val in freq.items():
    #     print(str(key), ':', str(val))
    # freq.plot(30)
    # plt.title(positive_option + " Word Frequencies")
    # ====================    THIS IS THE END OF TENDENTSI WORD  ANALYSIS =========================

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_png = buffer.getvalue()
    plot = base64.b64encode(image_png).decode()
    plots.append(plot)
    buffer.close()

    plt.clf()


    # PIE CHART
    # def get_color(selectedSentiment):
    #     colors = {'positif': 'lightblue', 'negatif': 'darkblue', 'netral': 'skyblue'}
    #     return [colors[sentiment] for sentiment in selectedSentiment]
        # elif sentiment == 'positif':
        #     return {'positif': 'darkblue', 'negatif': 'lightblue', 'netral': 'skyblue'}
        # elif sentiment == 'netral':
        #     return {'positif': 'lightblue', 'negatif': 'lightblue', 'netral': 'darkblue'}
        # if selectSentiment == 'Positive' or 'positive':
    # colors = {
    #     'positive': 'darkblue',
    #     'negative': 'lightblue',
    #     'neutral': 'skyblue'
    # }
    # colors = get_color(selectSentiment)

    pie_sentiment_counts = df.groupby('sentiment')['reviews'].count()
    plt.pie(pie_sentiment_counts.values, labels=pie_sentiment_counts.index, autopct='%1.2f%%')
            # colors=[colors[sentiment] for sentiment in pie_sentiment_counts.index])  # autopct='%1.2f%%'
    plt.title('Sentiment Analysis Results')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_png = buffer.getvalue()
    plot = base64.b64encode(image_png).decode()
    plots.append(plot)
    buffer.close()
    
    plt.clf()
    # ==============        visualizing data stop      =====================

    return plots, res_sentiment, most_common_words, words, frequencies    #result_positive, result_negative


    # for dataset in datasets:
    #     data = dataset['data']

    #     if dataset['type'] =='bar':
    #         sentiment_counts = df.groupby('sentiment')['reviews'].count()
    #         plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.2f%%')
    #         plt.title('Sentiment Analysis Results')
    #     elif dataset['type']== 'pie':
    #         tipe_subjectivity = df.groupby(['sentiment']).sum()['subjectivity']
    #         tipe_subjectivity.plot(kind='bar')
    #         plt.ylabel('Subjectivity')
    #         plt.xlabel('Sentiment')
    #         plt.title('Sentiment across Subjectivity')



    # sentiment_counts = df.groupby('sentiment')['reviews'].count()

    # plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.2f%%')
    # plt.title('Sentiment Analysis Results')

    # Visualization of Sentiment across Polarity
# the "Positive" sentiment has the most than the Negative and Neutral

    # tipe_subjectivity = df.groupby(['sentiment']).sum()['subjectivity']
    # tipe_subjectivity.plot(kind='bar')
    # plt.ylabel('Subjectivity')
    # plt.xlabel('Sentiment')
    # plt.title('Sentiment across Subjectivity')
    # plt.show()

    #     buffer = io.BytesIO()
    #     plt.savefig(buffer, format='png')
    #     buffer.seek(0)  # rewind the buffer to start of content

    #     # encode plot ke dalam base64
    #     image_png = buffer.getvalue()
    #     graph1 = base64.b64encode(image_png).decode()
    #     buffer.close()

    #     plt.clf()

    # return graph1, random_review


    # return res_fin
