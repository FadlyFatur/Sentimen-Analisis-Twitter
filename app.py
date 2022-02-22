from flask import Flask, request
from flask.templating import render_template
# import numpy as np
import pandas as pd
import sys
import tweepy
import re
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['FLASK_ENV'] = 'development'
app.config['FLASK_DEBUG'] = 1


@app.before_first_request
def _run_on_start():
    global tokenizer, data, model_BiGRU, model_GRU, apikey, apisecret, access_token, access_secret_token, api, kamus_slangword
    model_GRU = load_model('best_model_GRU.h5')
    model_BiGRU = load_model('best_model_BiGRU.h5')

    vocab_size = 15000
    oov_tok = "<OOV>"

    data = pd.read_csv(
        r'F:\kuliah\TA\WebApps\Test-1-flask\static\sentimen_data_cleaning_fase_4_v2 .csv')
    data = data[['text', 'sentimen']]
    print('data loaded', file=sys.stderr)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(data['text'])
    word_index = tokenizer.word_index
    total_vocab = len(word_index)
    print('Initializing complete!\n', total_vocab, file=sys.stderr)

    apikey = "SPi9E8mgxTlcWrLk3DwfhUiw2"
    apisecret = "JSe6pUKg55kSkKLlOrWS8Y3RIYUtYHWHbXeldqBhCSUNqLMyqL"
    access_token = "249944990-36gTQlbwkV9whN42ZYelMXANgPBo2VkoCepQ2qta"
    access_secret_token = "CPAU1TWYx0fefWzNjZlcHTe0rHFvkVwjtt5A3Qs2ngr6O"

    auth = tweepy.OAuthHandler(apikey, apisecret)
    auth.set_access_token(access_token, access_secret_token)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    print('Auth Tweepy Success!\n', file=sys.stderr)

    indo_slang = pd.read_csv(
        r'F:\kuliah\TA\WebApps\Test-1-flask\static\slang-indo.csv', delimiter=",")
    kamus_slangword = dict(indo_slang.values)
    print('Load slang Success!\n', file=sys.stderr)


def fetching_data(teks, user, total):
    search_words = teks + " " + user + \
        " -filter:retweets -#repost -repost -rt -is:retweet"
    print(search_words, file=sys.stderr)
    # date_since = "2020-06-10"
    tweets = tweepy.Cursor(api.search_tweets, q=search_words,
                           lang="id", tweet_mode="extended").items(total)
    return tweets


def filtering(text):
    # Make text lowercase
    text = text.lower()
    # remove new line/ enter
    text = re.sub('\n', ' ', text)
    # remove text in square brackets
    text = re.sub('\[.*?\]', '', text)
    # # remove text in angle brackets
    text = re.sub('\<.*?\>', '', text)
    # # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # # Remove @username
    text = re.sub('@[^\s]+', '', text)
    # # Remove hastag / #tagger
    text = re.sub(r'#([^\s]+)', '', text)
    # # remove punctuation / tanda baca
    text = re.sub('<.*?>+', '', text)
    # # remove words containing numbers
    text = re.sub(r'[0-9]', '', text)
    # #remove redudant ...
    text = re.sub(r"[,.;@#?!&$]+\ *", " ", text)
    # # remove more space
    text = re.sub(' +', ' ', text)
    return text


def repeatcharNormalize(text):
    huruf = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i in range(len(huruf)):
        charac_long = 5
        while charac_long >= 2:
            char = huruf[i]*charac_long
            text = text.replace(char, huruf[i])
            charac_long -= 1
    return text


def unslang(text):
    if text in kamus_slangword.keys():
        return kamus_slangword[text]
    else:
        return text


def hasil_unslang(kalimat, hasil_data=""):
    kalimat = kalimat.split()
    hasil = []
    # for loop to iterate over words array
    for kata in kalimat:
        filteredData = unslang(kata)
        # print(filteredData)
        hasil.append(filteredData)
        # print(hasil)
        hasil_data = ' '.join(hasil)

    return hasil_data


def predict_sentimen(text, model):
    trunc_type = 'post'
    padding_type = 'post'
    max_length = 100
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length,
                           padding=padding_type, truncating=trunc_type)
    if model == '1':
        model = model_GRU
        print('---------------------------------------------', file=sys.stderr)
        print('Model GRU Loaded\n', file=sys.stderr)
    else:
        model = model_BiGRU
        print('---------------------------------------------', file=sys.stderr)
        print('Model BiGRU Loaded\n', file=sys.stderr)

    score = model.predict([padded])[0]

    if score <= 0.5:
        sentimen_label = 'Negatif'
    else:
        sentimen_label = 'Positif'

    return (score, sentimen_label)


@ app.route('/')
def index():
    return render_template("index.html")


@ app.route('/real-time-test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        text = request.form['sentimen']
        model = request.form['model']

        print('---------------------------------------------', file=sys.stderr)
        print('Model : ', request.form['model'], '\n')

        score, label = predict_sentimen(text, model)
        return render_template("test.html", score=float(score), label=label, text=text, model=model)
    else:
        return render_template("test.html")


@ app.route('/sosmed-test', methods=['GET', 'POST'])
def sosmed():
    if request.method == 'POST':
        text = request.form['text']
        user = request.form['user']
        model = request.form['model']
        data_total = int(request.form['jml'])
        tweets = fetching_data(text, user, data_total)
        json_data = [tweet._json for tweet in tweets]
        print('Fetching Complete -----------------------------------', file=sys.stderr)

        df = pd.json_normalize(json_data)
        data_minimize = df[["full_text"]]
        data = data_minimize.rename(columns={'full_text': 'text'})
        Total_data = data.count()
        print('Convert to Pandas -----------------------------------', file=sys.stderr)

        data['Pre-text'] = data['text'].apply(lambda x: filtering(x))
        data['Pre-text'] = data['Pre-text'].apply(
            lambda x: repeatcharNormalize(x))
        data['Pre-text'] = data['Pre-text'].apply(lambda x: hasil_unslang(x))

        print('Pre-Processing Complete -----------------------------', file=sys.stderr)

        hasil = []
        score = []
        for text in data['Pre-text']:
            sc, hs = predict_sentimen(text, model)
            hasil.append(hs)
            score.append(sc)

        data["score"] = score
        data["hasil"] = hasil
        # data = data.drop('text', 1)
        print('Pre-Processing Complete -----------------------------', file=sys.stderr)

        neg, pos = data['hasil'].value_counts(
            normalize=True).mul(100).round(1).astype(str).tolist()

        return render_template("test_sosmed.html", total=Total_data,
                               tables2=[data.to_html(classes='table table-striped tables', header="true", index=False)], neg=neg, pos=pos)
    else:
        return render_template("test_sosmed.html")


if __name__ == "__main__":
    app.run(debug=True)
