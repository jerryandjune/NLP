# -*- coding: utf-8 -*-

# Copyright 2019 Arie Bregman
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
from flask import render_template, request, jsonify
from app.main import bp  # noqa
from app.models.job import Job
from app.models.Forms import NewsForm
from app.NewsData import NewsData
from app.sentence2vec import sentence2vec
import json

# @bp.route('/')
# def index():
#     """Main page route."""
#     button_text = "Add Job"
#     return render_template('Index.html', button_text=button_text)
#     # return render_template('main.html', button_text=button_text)


# @bp.route('/add_job')
# def add_job():
#     """Adds job4 to the database."""
#     new_job = Job(name='job4')
#     new_job.insert()
#     return ('', 204)

# about页面
@bp.route("/", methods=['Get', 'Post'])
def Index():
    newform = GetNews()
    if newform.validate_on_submit():
        pass
    return render_template("Index.html", form=newform)

# index页面
@bp.route("/About", methods=['Get'])
def About():
    return render_template("About.html")

# 获取摘要信息
@bp.route("/GetNewSummary", methods=['Post'])
def GetNewSummary():
    NewsTitle = request.values['NewsTitle']
    NewsContent = request.values['NewsContent']
    NewSummaryLength = int(request.values['NewSummaryLength'])

    news_title = '{}'.format(NewsTitle)
    news_content = '{}'.format(NewsContent)
    # todo
    summary = sentence2vec.get_summarize(news_content, news_title, weight = 0.7,top_n = NewSummaryLength)

    return jsonify({"result": summary})


@bp.route("/LoadData", methods=['GET'])
def LoadData():
    newform = GetNews()
    #d = newform.GetDict()
    #jsonret = json.dumps(newform.__dict__,ensure_ascii=False)
    return jsonify({"result": 
        {
        "NewsTitle":newform.NewsTitle,
        "NewsContent":newform.NewsContent,
        "NewSummaryLength":newform.NewSummaryLength 
        }
    })

def GetNews():
    newform = NewsData.GetNewData()
    return newform
