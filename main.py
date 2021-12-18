# importing all necessary packages
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import emoji
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

# download the english words corpus from nltk
nltk.download("words")
nltk_words = set(map(lambda x: x.lower(), nltk.corpus.words.words()))
nltk_words = nltk_words.union(
    set([
        "doge", "shib", "dogecoin", "shiba", "crypto", "com", 
        "guys", "hi", "hello", "lol", "has", "got"
        ]))

# reading the data into a dataframe and creating empty columns to 
# store the information produced at the later stage
df = pd.read_json("crypto_data.json")

df["message"] = [""]*len(df)
df["retain_message"] = [False]*len(df)
df["sentiment"] = [0]*len(df)

count_en = 0
count_non_en = 0
sia = SentimentIntensityAnalyzer()

print("\nMessage pre-processing and Sentiment Analysis starts")
for i in tqdm(range(0, len(df))):

    msg_meta = df.at[i, "messages"]
    msg = msg_meta.get("text")

    if type(msg) == list:
        # flattending a list by removing the dictionaries present inside
        msg = " ".join(
            list(map(lambda x: "" if type(x) == dict else str(x), msg))
            ).lower()
    else:
        # converting the message to lower case other wiae
        msg = str(msg.lower())

    # converting emojis into text so that the sentiment analyser can detect 
    # the sentiment in the emotion later on
    df.at[i, "message"] = emoji.demojize(msg, delimiters=("", ""))
    df.at[i, "date"] = msg_meta.get("date")[:10]

    # considering only those messages with shib or doge in them in the 
    # outer IF so that the data processed for language detection and 
    # sentiment analyser is filtered instead of the entire data
    if "shib" in msg or "doge" in msg:
        # removing all special characters
        msg = re.sub("[\!\@\#\$\%\^\&\*\_\-\+\=\;\:\<\>\,\.\?\/\|]", " ", msg)
        msg_split = set(msg.split(" "))
        # getting the intersection of the sentence with the nltk 
        # english word corpus created earlier
        msg_nltk_intersect = msg_split.intersection(nltk_words)
        # can use langdetect.detect(msg) == "en" as an alternative, 
        # but langdetect does not take into account the slang used in
        # most social medias these days
        # in my logic, if the intersection is greater than 1/3 the size of the
        # message, we consider it english
        if len(msg_nltk_intersect) >= 0.33*len(msg_split):
            df.at[i, "retain_message"] = True
            # computing the sentiment of the message by using NLTK VADER 
            # because it is built for usecases pertaining to messages on 
            # social media, and the function returns a compound score which is
            # in the range of -1 to 1 determining the sentiment of the message
            df.at[i, "sentiment"] = 100 * (
                sia.polarity_scores(msg).get("compound") + 1) / 2
            count_en+=1
        else:
            count_non_en+=1

print("Message pre-processing and Sentiment Analysis ends")
print("Number of English Shib/Doge Text Messages : ", count_en)
print("Number of Non-English Shib/Doge Text Messages : ", count_non_en)

# removing non shib/doge and non-english messages from the dataframe
df.drop(index = df[np.logical_not(df.retain_message)].index, inplace = True)
df.drop(columns = df.columns.difference(
    ["message", "date", "sentiment"]), 
    inplace = True)

df.sort_values(by="sentiment", ascending=False, inplace = True)
df = df.reset_index().drop(columns = "index")
# computing the date wise aggregate of the resultant dataframe
df_aggregate = df.groupby("date").apply(
    lambda x: pd.Series({
        "num_messages" : x.sentiment.count(), 
        "mean_sentiment" : round(x.sentiment.mean(), 2)})
        ).reset_index()

df_aggregate.to_csv("result_data.csv", index = False)
print("\nDay wise sentiment mean")
print(df_aggregate)

plt = px.line(
    df_aggregate, x="date", y="mean_sentiment", text = "mean_sentiment",
    title="Date Wise Sentiment mean of Crypto.com Telegram channel",
    markers=True, labels=dict(date="Date",mean_sentiment="Mean Sentiment (%)"))

plt.update_traces(textposition = "bottom right")

plt.show()