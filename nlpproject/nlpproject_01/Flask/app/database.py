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
import pymongo
from urllib import parse
from app.models.Config import Config

class DB(object):

    if Config.MongoDbAuth:
        URI = 'mongodb://{}:{}@{}:{}'.format(
            Config.MongoDbUsername,
            Config.MongoDbPassword,
            Config.MongoDbHost,
            Config.MongoDbPort)
    else :
        URI = 'mongodb://{}:{}'.format(Config.MongoDbHost,Config.MongoDbPort)

    @staticmethod
    def init():
        client = pymongo.MongoClient(DB.URI)
        DB.DATABASE = client['sample_app']

    @staticmethod
    def insert(collection, data):
        DB.DATABASE[collection].insert(data)

    @staticmethod
    def find_one(collection, query):
        return DB.DATABASE[collection].find_one(query)
