from boilerpipe.extract import Extractor
import re
import nltk
from nltk.metrics import association
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

class WebStatic:
    def __init__(self):
        self.URL = ''
        self.extractor = ''

    def setUrl(self, URL):
        self.URL = URL

    def getTextWeb(self):
        self.extractor = Extractor(extractor='KeepEverythingExtractor', url=self.URL)
        return self.extractor.getText()

    def getArticleText(self):
        self.extractor = Extractor(extractor='ArticleExtractor', url=self.URL)
        return self.extractor.getText()

    def getNews(self):
        self.extractor = Extractor(extractor='KeepEverythingExtractor', url=self.URL)
        buffer = list(self.extractor.getText().split(' '))
        buffer_two = []
        isnews = False
        pattern = '.\s\d\d.\d\d.\d{4}'
        for item in list(buffer):
            item = str(item).split()
            item = ' '.join(item)
            if re.search(pattern, item):
                isnews = True
                item = str(item).split(' ')
                buffer_two.append(item[0])
                item = item[1]
            if item == '':
                isnews = False
            if isnews:
                buffer_two.append(item)
        buffer_two.pop(0)
        pattern_year = '\d\d.\d\d.\d{4}'
        self.news = []
        newses = ''
        isnew = False
        for item in buffer_two:
            if re.search(pattern_year, item):
                newses = newses.replace('!', '').replace(',', '').replace('«', '').replace('»', '').replace(':',
                                                                                                            '').replace(
                    '–', ' ')
                self.news.append(newses)
                newses = ''
                isnew = True
                continue
            if isnew:
                newses = newses + '' + item
                isnew = False
            else:
                newses = newses + ' ' + item
        self.news.pop(0)
        return self.news

    def getRelevantNews(self):
        # Определите здесь свой запрос
        QUERY_TERMS = ['стол', 'кубка', 'регион']
        # получаем массив новостей
        self.news = self.getNews()
        # Textcollection определяет абстракции tf, idf и tf_idf,
        # поэтому нам не требуется определять свои версии
        tc = nltk.TextCollection(self.news)
        relevant = []
        for idx in range(len(self.news)):
            score = 1
            for term in [t.lower() for t in QUERY_TERMS]:
                score += tc.tf_idf(term, self.news[idx])
            if score > 0:
                relevant.append({'score': score, 'title': self.news[idx]})
        # Сортировать результаты по релевантности и выводим
        relevants = sorted(relevant, key=lambda p: p['score'], reverse=True)
        for post in relevants:
            print('{0}'.format(post['title']))
        return relevants

    def getCollocation(self):
        # Число искомых словосочетаний
        N = 10
        all_tokens = [token for post in self.news for token in post.lower().split()]
        for word in self.news:
            all_tokens.append(word.lower())
        finder = nltk.BigramCollocationFinder.from_words(all_tokens)
        finder.apply_freq_filter(2)
        finder.apply_word_filter(lambda w: w in nltk.corpus.stopwords.words('english'))
        scorer = association.BigramAssocMeasures.jaccard
        collocations = finder.nbest(scorer, N)
        for collocation in collocations:
            c = ' '.join(collocation)
            print(c)

    def getMatrixDiag(self):
        vector = TfidfVectorizer(analyzer='word', norm=None, use_idf=True, smooth_idf=True)
        tfIdf = vector.fit_transform(self.news)
        sim = cosine_similarity(tfIdf, tfIdf)
        newsList = []
        x=1
        for i in self.news:
            newsList.append(str(x))
            x=x+1
        simDf = pd.DataFrame(sim, index=sorted(newsList), columns=sorted(newsList))
        f = plt.figure(figsize=(19, 15))
        plt.matshow(simDf.corr(), fignum=f.number)
        plt.xticks(range(simDf.shape[1]), simDf.columns, fontsize=14, rotation=45)
        plt.yticks(range(simDf.shape[1]), simDf.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Косинусное сравнение новостей', fontsize=16);
        plt.show()
        print(simDf)

if __name__ == '__main__':
    OmSTU = WebStatic()
    OmSTU.setUrl("https://www.omgtu.ru/general_information/news/")
    print("Новости по релевантности с сайта ОмГТУ:")
    OmSTU.getRelevantNews()
    print()
    print("До 10 популярных словосочетаний:")
    OmSTU.getCollocation()
    OmSTU.getMatrixDiag()

    omstu_second = WebStatic()
    omstu_second.setUrl('https://www.omgtu.ru/l/p/?PAGEN_1=2')
    print("Новости по релевантности с сайта ОмГТУ(вторая ссылка):")
    omstu_second.getRelevantNews()
    print()
    print("До 10 популярных словосочетаний:")
    omstu_second.getCollocation()
    omstu_second.getMatrixDiag()

    omstu_second = WebStatic()
    omstu_second.setUrl('https://www.omgtu.ru/l/p/?PAGEN_1=3')
    print("Новости по релевантности с сайта ОмГТУ(третья ссылка):")
    omstu_second.getRelevantNews()
    print()
    print("До 10 популярных словосочетаний:")
    omstu_second.getCollocation()
    omstu_second.getMatrixDiag()

