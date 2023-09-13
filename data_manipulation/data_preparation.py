import string
from nltk.corpus import stopwords
import re
from pandas import read_csv, DataFrame

remove_punctuations = lambda input: ''.join([char for char in input if char not in string.punctuation])
remove_stop_words = lambda input: ' '.join(
    [word for word in input.split() if word.lower() not in stopwords.words('english')])


def remove_URL(text):
    text = str(text)
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_mentions(text):
    ment = re.compile(r"(@[A-Za-z0-9]+)")
    return ment.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def clean_tweet(tweet):
    tweet = remove_URL(tweet)
    tweet = remove_html(tweet)
    tweet = remove_mentions(tweet)
    tweet = remove_emoji(tweet)
    tweet = remove_punctuations(tweet)
    tweet = remove_stop_words(tweet)
    return tweet.strip().replace('  ', ' ')


PATH = '../input/train_labeled.csv'


def main():
    df = read_csv(PATH)

    output = []
    for item in df.values:
        cleaned = clean_tweet(item[0])
        item[0] = cleaned
        if len(cleaned) > 5:
            output.append(item)

    output = DataFrame(output)
    out = output.to_csv(index=False, header=True)

    output_file = PATH[:-4] + '_prepared.csv'
    with open(output_file, 'w') as f:
        f.write(out)


if __name__ == '__main__':
    main()
