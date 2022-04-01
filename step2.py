import contractions
import string
import re


def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    stopwords = [word.strip() for word in open('stopwords_en.txt')]
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


def main():
    text = "I read this book for the first time in 1987, and it's still one of my favorites!"
    clean_text(text)


if __name__ == '__main__':
    main()
