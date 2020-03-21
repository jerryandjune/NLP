# -*- coding: utf-8 -*-

#!/usr/bin/env python

from flask import Flask

from app.database import DB
from flask_bootstrap import Bootstrap
from app.models.job import Job
from app.models.Config import Config
from app.NewsData import NewsData
import pandas as pd
import os
from gensim.models import Word2Vec
from app.sentence2vec import sentence2vec

def create_app(new):
    app = Flask(__name__)
    bootstrap = Bootstrap(app)
    app.config.from_object(Config)

    #根据config启用mongodb
    if Config.MongoDbEnable:
        DB.init()
        for job_name in ['job1', 'job2', 'job3']:
            new_job = Job(name=job_name)
            new_job.insert()
            
    sentence2vec.init()
    NewsData.init()
    register_blueprints(app)
    return app

def register_blueprints(app):

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
