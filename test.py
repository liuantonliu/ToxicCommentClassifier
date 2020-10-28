import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)
# df = pd.read_csv("train_1.csv")
df = pd.read_csv("trainclean.csv")
df = pd.DataFrame(df.comment_text.apply(round1))
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(df.comment_text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = df.index
print(data_dtm)
df.to_csv("trainafter.csv")
# print(df)
