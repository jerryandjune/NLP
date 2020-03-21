# -*- coding: utf-8 -*- 

from flask_wtf import FlaskForm
from wtforms import StringField,IntegerField
from wtforms.validators import data_required,Length
from app.models.Config import Config

#新闻form
class NewsForm(FlaskForm):
    NewsTitle = StringField(u"标题")
    NewsContent = StringField(u"新闻",validators=[data_required("")])
    NewSummary = StringField(u"摘要")
    NewSummaryLength = IntegerField(u"摘要长度")
    
    def __init__(self,NewsTitle,NewsContent,NewSummary = ''):
        self.NewsTitle = NewsTitle
        self.NewsContent = NewsContent
        self.NewSummary = NewSummary
        self.NewSummaryLength = Config.SummaryLength


    def GetDict(self):
        return {
            'NewsTitle':self.NewsTitle,
            'NewsContent':self.NewsContent,
            'NewSummary':self.NewSummary,
            'NewSummaryLength':self.NewSummaryLength
        }