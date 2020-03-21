## 1. 项目目录

```
|-- nlpproject_01   
    |-- Flask
    	|-- app    	
            |-- main
                |-- __init__.py
                |-- routes.py
            |-- models
                |-- __init__.py
                |-- Config.py
                |-- Forms.py
                |-- job.py
            |-- Static
                |-- favicon.ico
                |-- Index.css
                |-- Index.js
                |-- message.js
                |-- sqlResult_1558435.csv【文件太大】        
            |-- templates
                |-- About.html
                |-- Base.html
                |-- Index.html
                |-- main.html
                |-- Navbar.html
            |-- __init__.py
            |-- database.py
            |-- NewsData.py
            |-- sentence2vec.py
            |-- requirements.txt
    |-- ModelTraining
    	|-- 01_parse_zhwiki_corpus.py
    	|-- 02_chinese_t2s.py
    	|-- 03_remove_english_blank.py
    	|-- 04_corpus_zhwiki_seg.py
    	|-- 05_corpus_news_seg.py
    	|-- 06_corpus_concat_seg.py
    	|-- 07_word2vec_train.py
    	|-- 08_FastText_train.py
    	|-- 09_sentence2vec.py
    	|-- 10_get_keywords.py
    	|-- NLP_Project_01 非监督文本自动摘要模型的构建.ipynb
    |-- README.md  	
```

## 2. Flask文件夹为部署项目的主要web文件，终端打开Flask文件夹，执行python -m flask run

## 3. ModelTraining文件夹为执行整个项目流程的py文件

## 4. 由于模型文件和新闻语料文件太大，不能上传，如果有执行项目，需要复制新闻文件和model文件到Flask\app\Static文件夹；配置Flask\app\models\Config.py中的新闻和model名字

## 5. 项目组员：梁俊杰、郝杰夫、李广宁、李珍