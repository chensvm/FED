import os
import re
import numpy as np


def readTrainingFileList():
    f = open('../training_articles/file.txt', 'w')
    for file in os.listdir("../training_articles"):
        if file == ".DS_Store":
            pass
        elif file == "file.txt":
            pass
        else:
            f.write(file+"\n")  # python will convert \n to os.linesep
    f.close()


def readTestingFileList():
    f = open('../testing_articles/file.txt', 'w')
    for file in os.listdir("../testing_articles"):
        if file == ".DS_Store":
            pass
        elif file == "file.txt":
            pass
        else:
            f.write(file + "\n")  # python will convert \n to os.linesep

    f.close()
#
# data = np.load('../testing_articles/20130101.npy')
# for line in data:
#     print line

# # coding=utf-8
# import requests
# from dateutil.parser import parse
# import re
# from bs4 import BeautifulSoup
#
# #http://www.reuters.com/news/archive/businessNews?view=page&page=1&pageSize=10
# base_url = "http://www.reuters.com/news/archive/businessNews?view=page&page="
# base_url2 = "&pageSize=10"
# url = "http://www.reuters.com"
#
# links = []
# for page in range(1, 3):
#     print("######page: "+ str(page))
#     ruters_r = requests.get(base_url + str(page) + base_url2)
#     ruters_soup = BeautifulSoup(ruters_r.text, 'html.parser')
#     finance = ruters_soup.findAll('div', {'class': 'story-content'})
#     for info in finance:
#         link = ""
#         try:
#             link = info.findAll('a', href=True)[0]
#             if link.get('href') != '#':
#                 links.append(url + link.get("href"))
#
#         except:
#             link = None
#
#     title = ""
#     time = ""
#     content = ""
#     test = True
#
#     for link in links:
#
#
#         news = requests.get(link)
#         single_news = BeautifulSoup(news.text, 'html.parser')
#
#         try:
#             mainpara = ""
#
#             title = single_news.find_all('h1', {'class': 'article-headline'})
#             #print title[0].text
#             time = single_news.find_all('span', {'class': 'timestamp'})
#             time = time[0].text #Fri Apr 7, 2017 | 5:43pm EDT
#             time = time.replace(' |', '') #Fri Apr 7, 2017 5:43pm EDT
#             dat = parse(time) #2017-04-07 17:43:00
#             dat = str(dat).replace(' ', '-').replace(':', '-')
#
#             content = single_news.find_all('p')
#             for elem in content:
#              elemcontent = ''.join(elem.text)
#              elemcontent = elemcontent.replace('/n', '')
#              mainpara += elemcontent
#
#             #print mainpara
#
#             file_path = "news/"+dat + ".txt"
#
#             with open(file_path, 'w') as textfile:
#
#                 print>>textfile, dat
#                 print>>textfile, title[0].text
#                 print>>textfile, mainpara
#
#
#                 textfile.close()
#         except:
#
#             continue

if __name__ == "__main__":
    readTrainingFileList()
    readTestingFileList()

