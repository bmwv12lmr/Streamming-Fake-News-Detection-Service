import bs4 as bs
import metadata_parser
import numpy as np
import os
import pandas as pd
import pickle
import requests
from fastai.text.all import *
from flask import Flask, render_template, jsonify, request


def getModel(file_name):
    model = None
    if os.path.exists(file_name):
        model = pickle.load(open(file_name, 'rb'))
    return model

def show(result):
  if result > 0.9:
      return ("pants-fire")
  elif result > 0.8:
      return ("false")
  elif result > 0.6:
      return ("barely-true")
  elif result > 0.5:
      return ("half-true")
  elif result > 0.3:
      return ("mostly-true")
  else:
      return ("true")

def mapLiarLiar(df):
    new_df = df.copy()
    new_df['label'].replace(['pants-fire'], int(0), regex=True, inplace=True)
    new_df['label'].replace(['false'], int(1), regex=True, inplace=True)
    new_df['label'].replace(['barely-true'], int(2), regex=True, inplace=True)
    new_df['label'].replace(['half-true'], int(3), regex=True, inplace=True)
    new_df['label'].replace(['mostly-true'], int(4), regex=True, inplace=True)
    new_df['label'].replace(['true'], int(5), regex=True, inplace=True)
    return new_df.label

app = Flask(__name__)

folder = "./"

class PROJECTDEADASS_UserBasedModel:
    def __init__(self, setCoeff=None, setIntercept=None):
        self.model1 = getModel(folder + "model1.pkl")
        self.model2 = getModel(folder + "model2.pkl")
        self.model3 = getModel(folder + "model3.pkl")
        self.setCoeff = setCoeff
        self.setIntercept = setIntercept

    def model1_train(self):
        real_df = pd.read_csv(
            "https://github.com/harshitkgupta/Fake-Profile-Detection-using-ML/raw/master/data/users.csv")
        fake_df = pd.read_csv(
            "https://github.com/harshitkgupta/Fake-Profile-Detection-using-ML/raw/master/data/fusers.csv")
        microFeatures = ["name"]
        real_mf_df = real_df[microFeatures].copy()
        real_mf_df["fake"] = False
        fake_mf_df = fake_df[microFeatures].copy()
        fake_mf_df["fake"] = True
        microFeatures.append("fake")
        data_mf_df = pd.concat([real_mf_df, fake_mf_df], ignore_index=True)
        data_mf_df.name = data_mf_df.name.apply(len)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import classification_report
        x_train, x_test, y_train, y_test = train_test_split(data_mf_df[microFeatures[:-1]],
                                                            data_mf_df[microFeatures[-1]],
                                                            test_size=0.2,
                                                            random_state=42)
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        self.model1 = GradientBoostingClassifier()
        self.model1.fit(x_train, y_train)
        self.model1_y_predict = self.model1.predict(x_test)
        print("Score:", self.model1.score(x_test, y_test))
        pickle.dump(self.model1, open(folder + "model1.pkl", 'wb'))

    def model1_predict(self, authorName):
        if self.model1 == None:
            self.model1_train()
        a = np.array([len(authorName)]).reshape(-1, 1)
        result = self.model1.predict(a)
        return result

    def model2_train(self):
        dataSet = pd.read_csv(untar_data(URLs.IMDB_SAMPLE) / "texts.csv");
        tdl = TextDataLoaders.from_df(dataSet, path=untar_data(URLs.IMDB_SAMPLE), text_col='text', label_col='label',
                                      valid_col='is_valid')
        self.model2 = text_classifier_learner(tdl, AWD_LSTM, drop_mult=0.5)
        self.model2.fine_tune(10)
        self.model2.predict('Very Good')
        pickle.dump(self.model2, open(folder + "model2.pkl", 'wb'))

    def model2_predict(self, text):
        if self.model2 == None:
            self.model2_train()
        return float(self.model2.predict(text)[2][1])

    def model3_train(self):
        dataTrain = pd.read_csv(folder + "train_processed.csv")
        dataTrain.label.unique()
        history_data_train = pd.DataFrame()
        history_data_train['barelytruecounts'] = dataTrain.barelytruecounts.copy()
        history_data_train['falsecounts'] = dataTrain.falsecounts.copy()
        history_data_train['halftruecounts'] = dataTrain.halftruecounts.copy()
        history_data_train['mostlytrueocunts'] = dataTrain.mostlytrueocunts.copy()
        history_data_train['pantsonfirecounts'] = dataTrain.pantsonfirecounts.copy()
        history_data_train['label'] = mapLiarLiar(dataTrain)
        history_data_train = history_data_train.dropna()
        history_data_train = history_data_train[~history_data_train.isin([np.nan, np.inf, -np.inf]).any(1)].astype(
            np.float64)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import classification_report
        x_train, x_test, y_train, y_test = train_test_split(history_data_train.drop(columns=['label']),
                                                            history_data_train.label,
                                                            test_size=0.2,
                                                            random_state=42)
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        self.model3 = GradientBoostingClassifier()
        self.model3.fit(x_train, y_train)
        self.model3_y_predict = self.model3.predict(x_test)
        print("Score:", self.model3.score(x_test, y_test))
        pickle.dump(self.model3, open(folder + "model3.pkl", 'wb'))

    def model3_predict(self, barelytruecounts, falsecounts, halftruecounts, mostlytrueocunts, pantsonfirecounts):
        if self.model3 == None:
            self.model3_train()
        history = np.array(
            [barelytruecounts, falsecounts, halftruecounts, mostlytrueocunts, pantsonfirecounts]).reshape(-1, 5)
        return self.model3.predict(history)[0] * 0.2

    def isFakeNews(self, text, authorName, barelytruecounts, falsecounts, halftruecounts, mostlytrueocunts,
                   pantsonfirecounts):

        model0_result = (1 - self.model1_predict(authorName))[0]
        model1_result = (1 - self.model2_predict(text))
        model2_result = (1 - self.model3_predict(barelytruecounts, falsecounts, halftruecounts, mostlytrueocunts,
                                                 pantsonfirecounts))

        w = [float(i) / sum(self.setCoeff) for i in self.setCoeff]
        prob = list()
        prob.append(w[0] * model0_result)
        prob.append(w[1] * model1_result)
        prob.append(w[2] * model2_result)
        probTotal = sum(prob[0:len(prob)])
        return probTotal + self.setIntercept

        return


def loadRSS(topicId='CAAqIggKIhxDQkFTRHdvSkwyMHZNRGxqTjNjd0VnSmxiaWdBUAE'):
    url = 'https://news.google.com/rss/topics/'+topicId
    resp = requests.get(url)
    with open('GoogleRSS.xml', 'wb') as f:
        f.write(resp.content)


def parseXML():
    import xml.etree.ElementTree as ET
    import pandas as pd

    xml_data = open('GoogleRSS.xml', 'r').read()  # Read file
    root = ET.XML(xml_data)  # Parse XML

    data = []
    cols = ['title', 'link', 'guid', 'pubDate', 'description', 'source']
    for i, root_1 in enumerate(root):
        for j, child in enumerate(root_1):
            if child.tag != 'item':
                continue
            data.append([subchild.text for subchild in child])
    df = pd.DataFrame(data)
    df.columns = cols
    cnn=df.loc[df['source'] == 'CNN'].reset_index(drop=True)
    for index, row in cnn.iterrows():
        row.link = metadata_parser.MetadataParser(row.link, search_head_only=False).url_actual
    fox = df.loc[df['source'] == 'Fox News'].reset_index(drop=True)
    for index, row in fox.iterrows():
        row.link = metadata_parser.MetadataParser(row.link, search_head_only=False).url_actual
    return cnn, fox

def url2CnnSpeaker(news_url_list):
    author_list=[]
    for url in news_url_list:
        page=metadata_parser.MetadataParser(url, search_head_only=False)
        authors = page.get_metadatas('author')[0]
        author = authors.replace(" and ", ",").split(',')[0].lower().replace(" ", "-")
        author_list.append(author)
    return author_list

def url2CnnTopic(news_url_list):
    topic_list=[]
    for url in news_url_list:
        page = metadata_parser.MetadataParser(url, search_head_only=False)
        topic = page.get_metadatas('section')
        if topic == None:
            topic = page.get_metadatas('meta-section')
        if topic == None:
            topic = page.get_metadatas('theme')
        topic_list.append(topic)
    return topic_list

def url2FoxSpeaker(news_url_list):
    author_list=[]
    for url in news_url_list:
        response = requests.get(url)
        soup = bs.BeautifulSoup(response.text,'html.parser')
        author = soup.find('div', class_='author-byline').text.replace("By","").replace("| Fox News","").replace("\n","").split(",")[0].strip().lower().replace(" ","-")
        author_list.append(author)
    return author_list

def url2FoxTopic(news_url_list):
    topic_list=[]
    for url in news_url_list:
        response = requests.get(url)
        soup = bs.BeautifulSoup(response.text,'html.parser')
        topic = soup.find('div', class_='eyebrow').text
        topic_list.append(topic)
    return topic_list


def get_result(model, raw, topic_list, author_list, history_list, result_list):
    for i in range(len(author_list)):
        result = model.isFakeNews(raw.title[i],
                                     author_list[i],
                                     history_list[i][0],
                                     history_list[i][1],
                                     history_list[i][2],
                                     history_list[i][3],
                                     history_list[i][4])
        result_list.append([raw.link[i], raw.title[i], topic_list[i], result, show(result)])


@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('index.html')


@app.route('/news', methods=['POST'])
def get_news():
    loadRSS()
    modelCoeff = [0.238196, 0.8816935, 0.20496854]
    modelIntercept = 0.03299003819074646
    cnn, fox = parseXML()
    cnn_author_list = url2CnnSpeaker(cnn.link.tolist())
    fox_author_list = url2FoxSpeaker(fox.link.tolist())
    cnn_topic_list = url2CnnTopic(cnn.link.tolist())
    fox_topic_list = url2FoxTopic(fox.link.tolist())
    name_history = load_csv()
    cnn_history = find_history(cnn_author_list, name_history)
    fox_history = find_history(fox_author_list, name_history)
    bigModel = PROJECTDEADASS_UserBasedModel(setCoeff=modelCoeff, setIntercept=modelIntercept)
    result = list()
    get_result(bigModel, cnn, cnn_topic_list, cnn_author_list, cnn_history, result)
    get_result(bigModel, fox, fox_topic_list, fox_author_list, fox_history, result)
    value = list()
    for i in range(len(result)):
        value.append({"Source":result[i][0], "Headline":result[i][1], "Topic":result[i][2], "Prob of Disinformation":result[i][3], "Liar Liar Label":result[i][4]})
    return {
        'news' : value
    }

def load_csv():
    input_data =pd.read_csv('train_processed.csv')
    df = input_data[["speaker", 'barelytruecounts', 'falsecounts', 'halftruecounts', 'mostlytrueocunts', 'pantsonfirecounts']]
    data = df.drop_duplicates(subset=['speaker']).reset_index()
    return data

def find_history(author_list, name_history):
    history = []
    for author in author_list:
        speak_info = name_history.loc[name_history['speaker'] == author]
        if len(speak_info) == 0:
            history.append([0,0,0,0,0])
        else:
            history.append([speak_info[0].barelytruecounts, speak_info[0].falsecounts, speak_info[0].halftruecounts, speak_info[0].mostlytrueocunts, speak_info[0].pantsonfirecounts])
    return history


if __name__ == '__main__':
    app.run(host='localhost', port=8080)
