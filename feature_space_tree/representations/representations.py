#!/usr/local/bin/python
# coding: utf-8

# Copyright (C) 2011-2012 FeatureSpaceTree Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# FeatureSpaceTree: Representations module
#
# Author: Adrian Pastor Lopez-Monroy <pastor@ccc.inaoep.mx>
# URL: <https://github.com/beiceman/FeatureSpaceTree>
#
# Language Technologies Lab,
# Department of Computer Science,
# Instituto Nacional de Astrofísica, Óptica y Electrónica
#
# For license information, see:
#  * The header of this file
#  * The LICENSE.TXT included in the project dir
# ==============================================================================

import math
import sys
import os
import random
import re
import time
import numpy
import scipy.sparse
import collections
import copy
import shelve
import glob
import codecs
import yaml
import json
from abc import ABCMeta, abstractmethod

import nltk
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from ..attributes.attr_config import FactoryTermLex
from ..attributes import virtuals
from _pyio import __metaclass__
from aptsources.distinfo import Template
from ..attributes.virtuals \
import FilterTermsVirtualGlobalProcessor, FilterTermsVirtualReProcessor

class bcolors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


class Util(object):
    
    
    # FIXME: Here there are a deging software error: As you can see the class
    # CorpusCategorizedFromCatMap still receiving kwargs_corpus['cat_pattern']
    # which is not necessary!!!. We have to rename the paremeters in the cons-
    # tructor of the class Corpus. This is because the latter class, forces...
    # (because it specyfies the parameters in the constructor) to all subclasses
    # to use these parameters. The solution is could be: Just Leave more generic
    # parameters (anyway subclass will specify them, or send it the complete
    # kwargs object. 
    @staticmethod
    def build_filtered_corpus_from_cat_map(categories, kwargs_corpus):
        if 'corpus_path' in kwargs_corpus and\
        'corpus_pattern' in kwargs_corpus and\
        'file_pattern' in kwargs_corpus and\
        'cat_pattern' in kwargs_corpus:
            corpus = CorpusCategorizedFromCatMap(categories,
                                       kwargs_corpus['corpus_path'],
                                       kwargs_corpus['corpus_pattern'],
                                       kwargs_corpus['file_pattern'],
                                       kwargs_corpus['cat_pattern'],
                                       kwargs_corpus['cat_map'],)
        else:
            corpus = CorpusCategorizedFromCatMap(categories, kwargs_corpus['corpus_path'], kwargs_corpus['cat_map'])

        corpus = Util.decorate_corpus(corpus, kwargs_corpus['filters_corpus'])
        return corpus
    
    @staticmethod
    def build_filtered_corpus(categories, kwargs_corpus):
        if 'corpus_path' in kwargs_corpus and\
        'corpus_pattern' in kwargs_corpus and\
        'file_pattern' in kwargs_corpus and\
        'cat_pattern' in kwargs_corpus:
            corpus = CorpusCategorized(categories,
                                       kwargs_corpus['corpus_path'],
                                       kwargs_corpus['corpus_pattern'],
                                       kwargs_corpus['file_pattern'],
                                       kwargs_corpus['cat_pattern'])
        else:
            corpus = CorpusCategorized(categories, kwargs_corpus['corpus_path'])

        corpus = Util.decorate_corpus(corpus, kwargs_corpus['filters_corpus'])
        return corpus

    @staticmethod
    def get_tuples_from_fdist(fdist):
        the_tuples = []
        for key in fdist.keys():
            #print str(type(key)) + u":>>>" + key
            elem_tuple = (key, fdist[key])
            the_tuples += [elem_tuple]

        return the_tuples

    @staticmethod
    def get_string_fancy_time(seconds, header):
        header = bcolors.OKGREEN + header + bcolors.ENDC
        string_time = \
        '''
        %s
        Seconds: %f
        Minutes: %f
        Hours:   %f
        '''

        string_time = (string_time % (header, seconds, seconds/60, seconds/60/60))

        return string_time

    @staticmethod
    def build_fancy_list_string(the_list, n_columns=3):

        fancy_list_string = ""
        column = 0
        i = 1
        for e in the_list:
            if column >= n_columns:
                fancy_list_string += "\n"
                column = 0
#            print i
#            print type(i)
#            print e
#            print type(e)
#            print e[0]
#            print type(e[0])
#            print e[1]
#            print type(e[1])
            fancy_list_string += "%-4d %-35s " % (i, e)
            column += 1
            i += 1

        return fancy_list_string
    
    @staticmethod
    def build_fancy_vocabulary(the_list):#, n_columns=3, enumerate=True):

        fancy_list_string = ""
        column = 0
        i = 1
        for e in the_list:
#            if column >= n_columns:
#                fancy_list_string += "\n"
#                column = 0
#            print i
#            print type(i)
#            print e
#            print type(e)
#            print e[0]
#            print type(e[0])
#            print e[1]
#            print type(e[1])
            #if enumerate:
            #    fancy_list_string += "%-6d " % i
        
            #fancy_list_string += "%-25s %-15s" % (str(e[0]), str(e[1]))
            #fancy_list_string += e[0] + " " +  str(e[1]) + "\n"
            fancy_list_string += str(e[1]) + " " +  e[0] + "\n"
#            column += 1
            i += 1

        return fancy_list_string

    @staticmethod
    def create_a_dir(a_new_dir):
        if not os.path.isdir(a_new_dir):
            os.mkdir(a_new_dir)

    @staticmethod
    def decorate_corpus(corpus, filters_corpus):

        for kwargs_filter_corpus in filters_corpus:
            filtered_corpus = \
            FactorySimpleFilterCorpus.create(kwargs_filter_corpus["type_filter_corpus"],
                                             kwargs_filter_corpus,
                                             corpus)
            corpus = filtered_corpus

#        corpus = \
#        FactorySimpleFilterCorpus.create(EnumFiltersCorpus.ORDER,
#                                         [],
#                                         corpus)
        return corpus


    @staticmethod
    def configure_categorized_corpus_from_cat_map(corpus_path,
                                                  cat_map,
                                     user_corpus_path=r'corpora/(.*/.*)',
                                     files_pattern=r'.+/.+',
                                     categories_pattern=r'(.+)/.+'):

        match = re.match(user_corpus_path,
                         corpus_path)

        corpus = LazyCorpusLoader(match.group(1),
                                  CategorizedPlaintextCorpusReader,
                                  files_pattern, cat_file=cat_map)
        return corpus

    @staticmethod
    def configure_categorized_corpus(corpus_path,
                                     user_corpus_path=r'corpora/(.*/.*)',
                                     files_pattern=r'.+/.+',
                                     categories_pattern=r'(.+)/.+'):

        match = re.match(user_corpus_path,
                         corpus_path)

        corpus = LazyCorpusLoader(match.group(1),
                                  CategorizedPlaintextCorpusReader,
                                  files_pattern, cat_pattern=categories_pattern)
        return corpus

#    @staticmethod
#    def configure_categorized_corpus(corpus_path):
#        match = re.match(r'~/nltk_data/corpora/(.*/.*)',
#                         corpus_path)
#
#        corpus = LazyCorpusLoader(match.group(1),
#                                  CategorizedPlaintextCorpusReader,
#                                  r'.+/.+', cat_pattern=r'(.+)/.+')
#        return corpus

#    @staticmethod
#    def configure_categorized_corpus_CHE(corpus_path):
#        match = re.match(r'~/nltk_data/corpora/(.+/.+/.+)',
#                         corpus_path)
#
#        corpus = LazyCorpusLoader(match.group(1),
#                                  CategorizedPlaintextCorpusReader,
#                                  r'.+', cat_pattern=r'.+_.+_.+_(.+)_.+')
#        return corpus
#
#    @staticmethod
#    def configure_categorized_corpus_Moon(corpus_path):
#        match = re.match(r'~/nltk_data/corpora/(.+/.+/.+)',
#                         corpus_path)
#
#        corpus = LazyCorpusLoader(match.group(1),
#                                  CategorizedPlaintextCorpusReader,
#                                  r'.+', cat_pattern=r'(.+)_.+')
#        return corpus

    @staticmethod
    def write_arrf(path, attributes, categories, matrix, name_files,  instance_categories, relation_name="my_relation", f_namefiles=True):
        '''
        This creates a arrf header string ready to be written into a weka's arff file
        '''
        string_arff = ''
        string_arff += ("@relation %s\n" % relation_name)
        # PAN13: print "Categories for the arff: ", str(categories)
        str_categories=", ".join(categories[:-1]) + ", " + categories[-1]

        i = 1
        for attr in attributes:
            matches = re.findall('[a-zA-Z0-9]+', attr)
            label = "_".join(matches)
            string_arff += ("@attribute %i_attr_%s numeric      %% real name: '%s'\n" % (i, label, attr))
            i += 1

        string_arff += ("@attribute categories {%s}\n" % str_categories)
        string_arff += ("@data\n")
        
        
        f_arff = open(path, 'w')
        f_arff.write(string_arff.encode('utf-8'))
        f_arff.close()
        
        if f_namefiles:
            f_namefiles = open(path + ".txt", 'w')        
        # Bad idea to other kinds of corpus, like CAT MAP CORPUS ---------------
        # categories_of_files = [re.match('(.+)/.+', name_file).group(1)
        #                        for name_file in name_files ]
        # ----------------------------------------------------------------------
        categories_of_files = instance_categories

        #categories_of_files = [re.match(".+_.+_.+_(.+)_.+", name_file).group(1)
        #                       for name_file in name_files ]
        c = re.compile('[a-zA-Z0-9]+')

# this is not faster!!!
#        i = 0
#        for i in range(len(name_files)):
#
#            row = matrix[i]
#            category_of_file = categories_of_files[i]
#            name_file = name_files[i]
        numpy.set_printoptions(precision=4, threshold='nan')
        
        f_arff = open(path, 'a')
        for (row,
             category_of_file,
             name_file) in zip(matrix,
                               categories_of_files,
                               name_files):

            # the last good: list_string_row = ['%.8f' % e for e in row]
            # the last good: list_string_row = ', '.join(list_string_row)
            list_string_row = ""

            #list_string_row = str(row)[1:-1]

            for e in row:
                list_string_row += str(e) + ", "

            valid_arff_name_file = "_".join(c.findall(name_file))

            # the last good: name_file = "_".join(re.findall('[a-zA-Z0-9]+',name_file))

            # the last good: string_arff += ('%s, %-25s %s\n' % (list_string_row, category_of_file, name_file))
            # the last good: string_arff += ('%s %-25s %s\n' % (list_string_row, category_of_file, name_file))
            
            f_arff.write((list_string_row + " " + category_of_file + "    " + valid_arff_name_file + "\n").encode('utf-8'))
            if f_namefiles:
                f_namefiles.write((name_file + " " + category_of_file + "\n").encode('utf-8'))
        
        f_arff.close()
        if f_namefiles:
            f_namefiles.close()

# the last good:
#        f_arff = open(path, 'w')
#        f_arff.write(string_arff.encode('utf-8'))
#        f_arff.close()

    @staticmethod
    def write_bin_matrix(path, attributes, categories, matrix, name_files, instance_categories, relation_name="my_relation"):
        
        # Bad idea to other kinds of corpus, like CAT MAP CORPUS ---------------
        # categories_of_files = [re.match('(.+)/.+', name_file).group(1)
        #                        for name_file in name_files ]
        # ----------------------------------------------------------------------
        categories_of_files = instance_categories
        numpy.save(path + "_cats.npy", numpy.array(categories_of_files))
        
        numpy.save(path + "_data.npy", matrix)

    @staticmethod
    def write_svmlib(path, attributes, categories, matrix, name_files, instance_categories, relation_name="my_relation"):
        '''
        This creates a arrf header string ready to be written into a weka's arff file
        BUGGY: I HAVE NOT VALIDATED THIS CODE
        '''  
        # BUG: I have not validated this code, so this could be buggy.     
        categories_of_files = []
        set_of_classes = []
        
#         for name_file in name_files:
#             class_name = re.match('(.+)/.+', name_file).group(1)
#             categories_of_files += [class_name]
#             
#             if class_name not in set_of_classes:
#                 set_of_classes += [class_name]

        for name_file, instance_category in zip(name_files, instance_categories):
            class_name = instance_category
            categories_of_files += [class_name]
            
            if class_name not in set_of_classes:
                set_of_classes += [class_name]
            
        k = 1    
        for e in set_of_classes:
            
            i = 0
            for class_n in categories_of_files:
                
                if class_n == e:
                    
                    categories_of_files[i] = k
                    
                i += 1
                    
            
            k += 1
                    
#        categories_of_files = [re.match('(.+)/.+', name_file).group(1)
#                               for name_file in name_files ]
        
        
        
        

        #categories_of_files = [re.match(".+_.+_.+_(.+)_.+", name_file).group(1)
        #                       for name_file in name_files ]
        c = re.compile('[a-zA-Z0-9]+')

# this is not faster!!!
#        i = 0
#        for i in range(len(name_files)):
#
#            row = matrix[i]
#            category_of_file = categories_of_files[i]
#            name_file = name_files[i]
        numpy.set_printoptions(precision=4, threshold='nan')
        
        f_arff = open(path, 'a')
        for (row,
             category_of_file,
             name_file) in zip(matrix,
                               categories_of_files,
                               name_files):

            # the last good: list_string_row = ['%.8f' % e for e in row]
            # the last good: list_string_row = ', '.join(list_string_row)
            list_string_row = ""

            #list_string_row = str(row)[1:-1]
            
            n = 1
            for e in row:
                list_string_row += str(n) + ":" + str(e) + " "
                n += 1
                
            
            name_file = "_".join(c.findall(name_file))

            # the last good: name_file = "_".join(re.findall('[a-zA-Z0-9]+',name_file))

            # the last good: string_arff += ('%s, %-25s %s\n' % (list_string_row, category_of_file, name_file))
            # the last good: string_arff += ('%s %-25s %s\n' % (list_string_row, category_of_file, name_file))
            
            f_arff.write((category_of_file + " " + list_string_row + "\n").encode('utf-8'))
        
        f_arff.close()

# the last good:
#        f_arff = open(path, 'w')
#        f_arff.write(string_arff.encode('utf-8'))
#        f_arff.close()

    @staticmethod
    def write_dict(path, attributes, categories, matrix, name_files, relation_name="my_relation"):
        '''
        This creates a arrf header string ready to be written into a weka's arff file
        '''
        string_arff = ''

        attr_processed = []
        i = 1
        for attr in attributes:
            matches = re.findall('[a-zA-Z0-9]+', attr)
            label = "_".join(matches)
            attr_processed += [("%i_attr_%s" % (i, label))]
            i += 1

        categories_of_files = [re.match('(.+)/.+', name_file).group(1)
                               for name_file in name_files ]

        #categories_of_files = [re.match(".+_.+_.+_(.+)_.+", name_file).group(1)
        #                       for name_file in name_files ]

        my_dict = shelve.open(path, protocol=2)
        for (row,
             category_of_file,
             name_file) in zip(matrix,
                               categories_of_files,
                               name_files):

            list_string_row = ['%.5f' % e for e in row]
            list_string_row = ', '.join(list_string_row)

            name_file_root = name_file.split("_")[0]
            if name_file_root in my_dict:
                my_dict[name_file_root][name_file] = ('%s, %-40s \n' % (list_string_row, category_of_file))
            else:
                my_dict[name_file_root] = {name_file: ('%s, %-40s \n' % (list_string_row, category_of_file))}

        my_dict["categories"] = categories
        my_dict["attributes"] = attr_processed
        my_dict.close()

    @staticmethod
    def get_list_of_files(path_specific_files):

        f = open(path_specific_files, 'r')

        list_of_files = []
        for line in f:
            list_of_files += [line.strip()]
            
        f.close()

        return list_of_files

    @staticmethod
    def getValidTokens_bak2(kwargs_term, set_vocabulary):
        factory_term_lex = FactoryTermLex()
        valid_tokens = []

        term = factory_term_lex.build_tokens(kwargs_term['type_term'], kwargs_term)
        valid_tokens += term.tokens
        #print "vocabulary: " + str(len(vocabulary))
        local_vocabulary = set_vocabulary & set(valid_tokens)
        #print "local vocabulary: " + str(len(local_vocabulary))
        valid_tokens = [token for token in valid_tokens if token in local_vocabulary]
        return valid_tokens
    
    @staticmethod
    def getValidTokens(kwargs_term, fdist):
        factory_term_lex = FactoryTermLex()
        valid_tokens = []

        term = factory_term_lex.build_tokens(kwargs_term['type_term'], kwargs_term)
        valid_tokens += term.tokens
        #print "vocabulary: " + str(len(vocabulary))
        #local_vocabulary = set(vocabulary) & set(valid_tokens)
        #print "local vocabulary: " + str(len(local_vocabulary))
        valid_tokens = [token for token in valid_tokens if token in fdist]
        return valid_tokens
    
    @staticmethod
    def getValidTokens_bak(kwargs_term, vocabulary):
        factory_term_lex = FactoryTermLex()
        valid_tokens = []

        term = factory_term_lex.build_tokens(kwargs_term['type_term'], kwargs_term)
        valid_tokens += term.tokens
        #print "vocabulary: " + str(len(vocabulary))
        local_vocabulary = set(vocabulary) & set(valid_tokens)
        #print "local vocabulary: " + str(len(local_vocabulary))
        valid_tokens = [token for token in valid_tokens if token in local_vocabulary]
        return valid_tokens


class EnumFiltersCorpus(object):

    (SPECIFIC_FILES,
     IMBALANCE,
     FULL,
     STRATIFIED_CROSS_FOLD) = range(4)


class FactorySimpleFilterCorpus(object):

    @staticmethod
    def create(option, kwargs, corpus):

        option = eval(option)
        if option == EnumFiltersCorpus.SPECIFIC_FILES:
            if 'file_of_specific_paths' in kwargs:
                specific_file_paths = Util.get_list_of_files(kwargs['file_of_specific_paths'])
            return SpecificFilesCorpus(corpus, specific_file_paths)

        if option == EnumFiltersCorpus.IMBALANCE:
            return ImbalancedFilesCorpus(corpus, kwargs['imbalance'])

        if option == EnumFiltersCorpus.FULL:
            return FullFilesCorpus(corpus)
        
        if option == EnumFiltersCorpus.STRATIFIED_CROSS_FOLD:
            return StratifiedCrossFoldFilesCorpus(corpus, 
                                                  kwargs['n_folds'], 
                                                  kwargs['target_fold'], 
                                                  kwargs['mode'], 
                                                  kwargs['seed'])


class Corpus(object):

    __metaclass__ = ABCMeta

    def __init__(self,
                 categories,
                 corpus_path,
                 user_corpus_path=r'corpora/(.*/.*)',
                 files_pattern=r'.+/.+',
                 categories_pattern=r'(.+)/.+'):

        self._categories = categories
        self._corpus_path = corpus_path

        self.user_corpus_path = user_corpus_path
        self.files_pattern = files_pattern
        self.categories_pattern = categories_pattern

        self.build_corpus(self._corpus_path,
                          self.user_corpus_path,
                          self.files_pattern,
                          self.categories_pattern)

    def get_categories(self):
        return self._categories

    def get_path(self):
        return self._corpus_path

    @abstractmethod
    def get_corpus(self):
        pass

    @abstractmethod
    def build_corpus(self, corpus_path, user_corpus_path, files_pattern, categories_pattern):
        pass

    @abstractmethod
    def get_docs(self):
        pass

class CorpusCategorized(Corpus):

    def __init__(self,
                 categories,
                 corpus_path,
                 user_corpus_path=r'corpora/(.*/.*)',
                 files_pattern=r'.+/.+',
                 categories_pattern=r'(.+)/.+'):

        super(CorpusCategorized, self).__init__(categories,
                                                corpus_path,
                                                user_corpus_path,
                                                files_pattern,
                                                categories_pattern)


    def get_corpus(self):
        return self._corpus

    def build_corpus(self, corpus_path, user_corpus_path, files_pattern, categories_pattern):
        self._corpus = Util.configure_categorized_corpus(corpus_path, user_corpus_path, files_pattern, categories_pattern)
        self.__docs = self._corpus.fileids(categories = self._categories)

        # DEBUG: Uncomment if you want to see the selected documents
        # print "CORPUS_PATH: " + self._corpus_path
        # print "DOCUMENTS: " + str(self.__docs)
        # print "CATEGORIES: " + str(self._categories)

    def get_docs(self):
        return self.__docs
    

class CorpusCategorizedFromCatMap(Corpus):

    def __init__(self,
                 categories,
                 corpus_path,
                 cat_map,
                 user_corpus_path=r'corpora/(.*/.*)',
                 files_pattern=r'.+/.+',
                 categories_pattern=r'(.+)/.+'):
        
        # FIXME:
        # BUG:
        # This is not exactly a bug, but it is a possible source of bug.
        # The class Corpus, in its constructor calls to build_corpus (it is a Template). 
        # So, since the super class calls the method, AND because python is interpreted
        # This class has not created the self.__cat_map attribute if it is not
        # called before the super....__init__ 
        self.__cat_map = cat_map
        # ----------------------------------------------------------------------
        
        super(CorpusCategorizedFromCatMap, self).__init__(categories,
                                                corpus_path,
                                                user_corpus_path,
                                                files_pattern,
                                                categories_pattern)
        
        
        
        

    def get_corpus(self):
        return self._corpus

    def build_corpus(self, corpus_path, user_corpus_path, files_pattern, categories_pattern):
        self._corpus = Util.configure_categorized_corpus_from_cat_map(corpus_path, self.__cat_map, user_corpus_path, files_pattern)
        self.__docs = self._corpus.fileids(categories = self._categories)

        # DEBUG: Uncomment if you want to see the selected documents
        # print "CORPUS_PATH: " + self._corpus_path
        # print "DOCUMENTS: " + str(self.__docs)
        # print "CATEGORIES: " + str(self._categories)

    def get_docs(self):
        return self.__docs


class FilterCorpus(Corpus):

    def __init__(self, corpus):
        self._corpus = corpus

    def get_categories(self):
        return self._corpus.get_categories()

    def get_path(self):
        return self._corpus.get_path()

    def get_corpus(self):
        return self._corpus.get_corpus()

    def build_corpus(self, corpus_path, user_corpus_path, files_pattern, categories_pattern):
        self._corpus.build_corpus(corpus_path, user_corpus_path, files_pattern, categories_pattern)

    @abstractmethod
    def get_docs(self):
        return self._corpus.get_docs()


class SpecificFilesCorpus(FilterCorpus):

    def __init__(self, corpus, list_specific_files):
        super(SpecificFilesCorpus, self).__init__(corpus)
        self.list_specific_files = list_specific_files

    def get_docs(self):

        old_train_docs = self._corpus.get_docs()
        # print "DEBUGGING: " + str(old_train_docs)
        for train_doc in self.list_specific_files:

            if train_doc not in old_train_docs:
                print("ERROR: file %s was not found in the train corpus." % train_doc)
                return None
            #else:
            #    print "DEBUGGING :) ...", train_doc

        self.list_specific_files.sort()
        return self.list_specific_files


class ImbalancedFilesCorpus(FilterCorpus):

    def __init__(self, corpus, list_imbalance):
        super(ImbalancedFilesCorpus, self).__init__(corpus)
        self.list_imbalance = list_imbalance

    def get_docs(self):
        old_train_docs = self._corpus.get_docs()
        docs = []

        for (cat, imbal) in zip(self.get_categories(), self.list_imbalance):
            author_file_list = self.get_corpus().fileids(categories=[cat])
            author_file_list = list(set(author_file_list) & set(old_train_docs))
            random.shuffle(author_file_list)
            docs += author_file_list[:imbal]

        docs.sort()
        return docs


class FullFilesCorpus(FilterCorpus):

    def __init__(self, corpus):
        super(FullFilesCorpus, self).__init__(corpus)

    def get_docs(self):
        old_train_docs = self._corpus.get_docs()
        docs = []

        for cat in self.get_categories():
            author_file_list = self.get_corpus().fileids(categories=[cat])
            author_file_list = list(set(author_file_list) & set(old_train_docs))
            docs += author_file_list

        docs.sort()
        # PAN13: print docs
        return docs
    

class StratifiedCrossFoldFilesCorpus(FilterCorpus):
    """This is BUGGY I have not tested yet!!!
    """

    def __init__(self, corpus, n_folds, target_fold, mode, seed):
        """ 
        mode is a string that could be: "train" or "test"
        """
        super(StratifiedCrossFoldFilesCorpus, self).__init__(corpus)
        self.n_folds = n_folds
        self.mode = mode
        self.seed = seed
        self.target_fold = target_fold

    def get_docs(self):
        old_train_docs = self._corpus.get_docs()
        docs = []

        for cat in self.get_categories():
            author_file_list = self.get_corpus().fileids(categories=[cat])
            author_file_list = list(set(author_file_list) & set(old_train_docs))
            
            
            author_file_list = sorted(author_file_list)            
            author_folds = [ e % self.n_folds for e in range(len(author_file_list))]
            
            random.seed(self.seed)
            random.shuffle(author_folds)
            
            selected_author_files_list = []
            for (author_file, author_fold) in zip(author_file_list, author_folds):
                
                if self.mode.strip() == "train":                    
                    if author_fold != self.target_fold:
                        selected_author_files_list += [author_file]                        
                elif self.mode.strip() == "test":                    
                    if author_fold == self.target_fold:
                        selected_author_files_list += [author_file]
                else:
                    print "Error, you must specify the mode in str: train or test"
                    sys.exit()
            
            docs += selected_author_files_list

        docs.sort()
        # PAN13: print docs
        return docs


class FactoryInfoClasses(object):

    @staticmethod
    def crear(authors, corpus, token_type, kwargs, fdist,
              corpus_file_list, tokens_path, style="NORMAL"):

        if style == "NORMAL":
            return VirtualCategoriesHolder(authors, corpus, token_type, kwargs, fdist,
                               corpus_file_list, tokens_path)

            
class TransformedDict(collections.MutableMapping):
    """A dictionary which applies an arbitrary key-altering function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs)) # use the free update to set keys
        
    def set_kwargs_terms(self, kwargs_terms):
        self._kwargs_terms = kwargs_terms
        
    def set_corpus(self, corpus):
        self._corpus = corpus
    
    def set_fdist(self, fdist):
        self._fdist = fdist

    def __getitem__(self, key):            
        #return self.store[self.__keytransform__(key)]
        file_path = self.store[self.__keytransform__(key)]
        list_file_tokens_combined = []
        for kwargs_term in self._kwargs_terms:
            kwargs_term['string'] = ''
            kwargs_term['source'] = [file_path]
            kwargs_term['corpus'] = self._corpus

            list_file_tokens_combined += \
            Util.getValidTokens(kwargs_term, self._fdist)
            
        # PAN13: print file_path,": ",list_file_tokens_combined
            
        return list_file_tokens_combined     

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
    

#class MyTransformedDict(TransformedDict):
#    
#    def __init__(self):
#        super(MyTransformedDict, self).__init__()
#    
#    def set_kwargs_terms(self, kwargs_terms):
#        self._kwargs_terms = kwargs_terms
#        
#    def set_corpus(self, corpus):
#        self._corpus = corpus
#    
#    def set_vocabulario(self, vocabulary):
#        self._vocabulary = vocabulary
#    
#    def __get_item__(self, key):
#        file_path = super(MyTransformedDict, self).__get_item__(key)
#        
#        list_file_tokens_combined = []
#        for kwargs_term in self._kwargs_terms:
#            kwargs_term['string'] = ''
#            kwargs_term['source'] = [file_path]
#            kwargs_term['corpus'] = self._corpus
#
#            list_file_tokens_combined += \
#            Util.getValidTokens(kwargs_term, self._vocabulary)
#            
#        print file_path,": ",list_file_tokens_combined
#            
#        return list_file_tokens_combined     
        

class VirtualCategory(object):

    def __init__(self, author, cat_file_list, dic_file_tokens):
        self.author = author        
        self.cat_file_list = cat_file_list
        self.dic_file_tokens = dic_file_tokens
        
class VirtualCategoriesHolder(object):
    '''
    The purpouse of this class is to retain all the posible information about
    each category.
    Note how this is useful when you want to see things like: the terms and its
    distribution in each category, or in each file, either in the train corpus
    or in the test corpus. This can give clues about why or why not things work.
    '''

    def __init__(self, categories, corpus, kwargs_terms, fdist,
                 corpus_file_list):

        # DEBUG: print "BUILDING VIRTUALS..."
        # this is a dictionary that will contain all posible information about
        # each category.
        self.virtual_categories = {}
        # ----------------------------------------------------------------------

        #kwargs_terms["corpus"] = corpus

        for kwargs_term in kwargs_terms:
                kwargs_term['string'] = ''
                kwargs_term['corpus'] = corpus
                kwargs_term['source'] = corpus_file_list

        for category in categories:

            cat_file_list = corpus.fileids(categories=[category])
            cat_file_list = [f for f in cat_file_list if f in corpus_file_list]

            # ------------------------------------------------------------------
            # INSPECT: This is to handle the case where exists several files
            # with the same name through the different categories. For example,
            # inside the dir "Juan" exists a file named "nota_1.txt" and inside
            # the dir "Jose" also exists a file named "nota_1.txt".
            #
            # cat_file_list = list(set(cat_file_list))
            #
            # ------------------------------------------------------------------
            # SOLVED: Does not matter because the name of the files in the list
            # are of the form "Juan/nota_1.txt" and "Jose/nota_1.txt" :D
            # ------------------------------------------------------------------

            cat_file_list.sort()

            # ------------------------------------------------------------------
            # Initialize the custom dictionary
            dic_file_tokens = TransformedDict()
            dic_file_tokens.set_kwargs_terms(kwargs_terms)
            dic_file_tokens.set_corpus(corpus)
            dic_file_tokens.set_fdist(fdist)
            # ------------------------------------------------------------------
            
            for author_file in cat_file_list:

                dic_file_tokens[author_file] = author_file
#                print "**********************************5555"
#                print author_file
#                print dic_file_fd[author_file]
#                print "**********************************5555"

                # DEBUG: print "TOKENS: ", dic_file_tokens[author_file]
                
            # Finally constructs the object that hold all our useful information
            self.virtual_categories[category] = \
            VirtualCategory(category,
                            cat_file_list,
                            dic_file_tokens)
            # ------------------------------------------------------------------

#            f_tokens = open(tokens_path, 'a')
#            f_tokens.write("\n" + category + '==============================\n');
#            temp_vocab_cat = []
#            # nltk sobreescribe keys() para ordenar
#            for item in fd_vocabulary_cat.items():
#                temp_vocab_cat += [str(item)]
#            #Utilities.writeFancyList(f_tokens, temp_vocab_cat)
#            f_tokens.close()
        
        
class VirtualCategory_bak(object):

    def __init__(self, author, num_tokens_cat, fd_vocabulary_cat,
                 cat_file_list, dic_file_tokens, dic_file_fd):
        self.author = author
        self.num_tokens_cat = num_tokens_cat
        self.fd_vocabulary_cat = fd_vocabulary_cat
        self.cat_file_list = cat_file_list
        self.dic_file_tokens = dic_file_tokens
        self.dic_file_fd = dic_file_fd


class VirtualCategoriesHolder_bak(object):
    '''
    The purpouse of this class is to retain all the posible information about
    each category.
    Note how this is useful when you want to see things like: the terms and its
    distribution in each category, or in each file, either in the train corpus
    or in the test corpus. This can give clues about why or why not things work.
    '''

    def __init__(self, categories, corpus, kwargs_terms, vocabulary,
                 corpus_file_list):

        # this is a dictionary that will contain all posible information about
        # each category.
        self.virtual_categories = {}
        # ----------------------------------------------------------------------

        #kwargs_terms["corpus"] = corpus

        for kwargs_term in kwargs_terms:
                kwargs_term['string'] = ''
                kwargs_term['corpus'] = corpus
                kwargs_term['source'] = corpus_file_list

        for category in categories:

            cat_file_list = corpus.fileids(categories=[category])
            cat_file_list = [f for f in cat_file_list if f in corpus_file_list]

            # ------------------------------------------------------------------
            # INSPECT: This is to handle the case where exists several files
            # with the same name through the different categories. For example,
            # inside the dir "Juan" exists a file named "nota_1.txt" and inside
            # the dir "Jose" also exists a file named "nota_1.txt".
            #
            # cat_file_list = list(set(cat_file_list))
            #
            # ------------------------------------------------------------------
            # SOLVED: Does not matter because the name of the files in the list
            # are of the form "Juan/nota_1.txt" and "Jose/nota_1.txt" :D
            # ------------------------------------------------------------------

            cat_file_list.sort()

            tokens_cat  = []
            for kwargs_term in kwargs_terms:
                kwargs_term['string'] = ''
                kwargs_term['source'] = cat_file_list
                kwargs_term['corpus'] = corpus
                tokens_cat += self.getValidTokens(kwargs_term, vocabulary)

            num_tokens_cat = len(tokens_cat)

            fd_vocabulary_cat = nltk.FreqDist(tokens_cat)

            dic_file_tokens = {}
            dic_file_fd = {}
            for author_file in cat_file_list:

                list_file_tokens_combined = []
                for kwargs_term in kwargs_terms:
                    kwargs_term['string'] = ''
                    kwargs_term['source'] = [author_file]
                    kwargs_term['corpus'] = corpus

                    list_file_tokens_combined += \
                    self.getValidTokens(kwargs_term, vocabulary)


                dic_file_tokens[author_file] = list_file_tokens_combined
                dic_file_fd[author_file] = \
                nltk.FreqDist(dic_file_tokens[author_file])
#                print "**********************************5555"
#                print author_file
#                print dic_file_fd[author_file]
#                print "**********************************5555"

            # Finally constructs the object that hold all our useful information
            self.virtual_categories[category] = \
            VirtualCategory(category,
                            num_tokens_cat,
                            fd_vocabulary_cat,
                            cat_file_list,
                            dic_file_tokens,
                            dic_file_fd)
            # ------------------------------------------------------------------

#            f_tokens = open(tokens_path, 'a')
#            f_tokens.write("\n" + category + '==============================\n');
#            temp_vocab_cat = []
#            # nltk sobreescribe keys() para ordenar
#            for item in fd_vocabulary_cat.items():
#                temp_vocab_cat += [str(item)]
#            #Utilities.writeFancyList(f_tokens, temp_vocab_cat)
#            f_tokens.close()

    def getValidTokens(self, kwargs_term, vocabulary):
        factory_term_lex = FactoryTermLex()
        valid_tokens = []

        term = factory_term_lex.build_tokens(kwargs_term['type_term'], kwargs_term)
        valid_tokens += term.tokens
        #print "vocabulary: " + str(len(vocabulary))
        local_vocabulary = set(vocabulary) & set(valid_tokens)
        #print "local vocabulary: " + str(len(local_vocabulary))
        valid_tokens = [token for token in valid_tokens if token in local_vocabulary]
        return valid_tokens


class CorpusObject(object):

    __metaclass__ = ABCMeta

    def __init__(self, categories, kwargs_corpus):
        self.categories = categories
        self.kwargs_corpus = kwargs_corpus

        template_option = kwargs_corpus['type_corpus']

        self.template_constructor = FactoryCommonCorpusTemplate().build_corpus(template_option)

        # PAN13: print self.template_constructor

        self.filtered_train_corpus = None
        self.filtered_test_corpus = None

        self.build_corpus()

    def build_corpus(self):
        self.template_constructor.calc_corpus(self)


class EnumCommonTemplate(object):
    (UNIQUE,
     TRAIN_TEST,
     TRAIN_TEST_FROM_CAT_MAP) = range(3)


class FactoryCorpusTemplate(object):

    def build_corpus(self, option):
        option = eval(option)
        return self.create(option)

    @abstractmethod
    def create(self, option):
        pass


class FactoryCommonCorpusTemplate(FactoryCorpusTemplate):

    def create(self, option):
        if option == EnumCommonTemplate.UNIQUE:
            return UniqueCorpusTemplate()
        elif option == EnumCommonTemplate.TRAIN_TEST:
            return TrainTestCorpusTemplate()
        elif option == EnumCommonTemplate.TRAIN_TEST_FROM_CAT_MAP:
            return TrainTestCorpusFromCatMapTemplate()


class CorpusTemplate(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def calc_corpus(self, corpus_object):
        pass


class UniqueCorpusTemplate(CorpusTemplate):

    def calc_corpus(self, corpus_object):
        corpus_object.filtered_train_corpus = \
        Util.build_filtered_corpus(corpus_object.categories,
                                            corpus_object.kwargs_corpus['unique_corpus'])

        corpus_object.filtered_test_corpus = \
        Util.build_filtered_corpus(corpus_object.categories,
                                            corpus_object.kwargs_corpus['null_corpus'])


class TrainTestCorpusTemplate(CorpusTemplate):

    def calc_corpus(self, corpus_object):
        corpus_object.filtered_train_corpus = \
        Util.build_filtered_corpus(corpus_object.categories,
                                            corpus_object.kwargs_corpus['train_corpus'])

        corpus_object.filtered_test_corpus = \
        Util.build_filtered_corpus(corpus_object.categories,
                                            corpus_object.kwargs_corpus['test_corpus'])
        

class TrainTestCorpusFromCatMapTemplate(CorpusTemplate):

    def calc_corpus(self, corpus_object):
        corpus_object.filtered_train_corpus = \
        Util.build_filtered_corpus_from_cat_map(corpus_object.categories,
                                            corpus_object.kwargs_corpus['train_corpus'])

        corpus_object.filtered_test_corpus = \
        Util.build_filtered_corpus_from_cat_map(corpus_object.categories,
                                            corpus_object.kwargs_corpus['test_corpus'])


class ConfigBaseAdvanced(object):

    def __init__(self, kwargs_config_base, global_kwargs_list):

        self.experiment_name = kwargs_config_base['experiment_name']
        self.experiment_base_path = kwargs_config_base['experiment_base_path']
        self.categories = kwargs_config_base['categories']
        self.processing_option = kwargs_config_base['processing_option']
        self.global_kwargs_list = global_kwargs_list

        self.corpus_object = \
        CorpusObject(self.categories,
                     kwargs_config_base['corpus'])

        self.__filtered_train_corpus = self.corpus_object.filtered_train_corpus

        self.__filtered_test_corpus = self.corpus_object.filtered_test_corpus
        
        # ======================================================================
        # the next two objects are the raw corpus!!!, it means that if you get
        # the documents or categories from this objects,...then ALL 
        # the documents will be returned. This means, that all the filters is 
        # not working on this objects.
        
        # This objects, are necesaries because in fact whe get all the things as objects, 
        # (files, categories, etc) from this objects (train_corpus and test corpus), but
        # since all the resources are returned, whe constantly filter things, using
        # le lists  self.corpus_file_list_train, and the self.categories. Which
        # are in fact the thing selected by the users!!!.

        self.train_corpus = self.__filtered_train_corpus.get_corpus()
        self.test_corpus = self.__filtered_test_corpus.get_corpus()
        # print "DEBUGGING X: ", self.train_corpus.categories()
        # ======================================================================
        
        # ======================================================================        
        # The following lists help us to filter all the stuff returned by the
        # train_corpus and test_corpus objects.

        self.corpus_file_list_train = self.__filtered_train_corpus.get_docs()
        self.corpus_file_list_test = self.__filtered_test_corpus.get_docs()
        # ======================================================================

        Util.create_a_dir(self.experiment_base_path)
        Util.create_a_dir(self.experiment_base_path + "/" + self.experiment_name)


class ConfigBase(object):

    def __init__(self, kwargs_config_base, global_kwargs_list):

        self.experiment_name = kwargs_config_base['experiment_name']
        self.experiment_base_path = kwargs_config_base['experiment_base_path']
        self.categories = kwargs_config_base['categories']
        self.processing_option = kwargs_config_base['processing_option']
        self.global_kwargs_list = global_kwargs_list

        self.__filtered_train_corpus = \
        self.__build_filtered_corpus(self.categories,
                                     kwargs_config_base['train_corpus'])

        self.__filtered_test_corpus = \
        self.__build_filtered_corpus(self.categories,
                                     kwargs_config_base['test_corpus'])

        self.train_corpus = self.__filtered_train_corpus.get_corpus()
        self.test_corpus = self.__filtered_test_corpus.get_corpus()

        self.corpus_file_list_train = self.__filtered_train_corpus.get_docs()
        self.corpus_file_list_test = self.__filtered_test_corpus.get_docs()

        Util.create_a_dir(self.experiment_base_path)
        Util.create_a_dir(self.experiment_base_path + "/" + self.experiment_name)

    def __build_filtered_corpus(self, categories, kwargs_corpus):
        if 'corpus_path' in kwargs_corpus and\
        'corpus_pattern' in kwargs_corpus and\
        'file_pattern' in kwargs_corpus and\
        'cat_pattern' in kwargs_corpus:
            corpus = CorpusCategorized(categories,
                                       kwargs_corpus['corpus_path'],
                                       kwargs_corpus['corpus_pattern'],
                                       kwargs_corpus['file_pattern'],
                                       kwargs_corpus['cat_pattern'])
        else:
            corpus = CorpusCategorized(categories, kwargs_corpus['corpus_path'])

        corpus = Util.decorate_corpus(corpus, kwargs_corpus['filters_corpus'])
        return corpus


class EnumRepresentation(object):

    (BOW,
     CSA) = range(2)

class AttributeHeader(object):

    _metaclass_ = ABCMeta

    def __init__(self, fdist, vocabulary, categories):
        self._fdist = fdist
        self._vocabulary = vocabulary
        self._categories = categories

    @abstractmethod
    def get_attributes(self):
        pass


class AttributeHeaderBOW(AttributeHeader):

    def __init_(self, fdist, vocabulary, categories):
        super(AttributeHeaderBOW, self).__init__(fdist, vocabulary, categories)

    def get_attributes(self):
        return self._vocabulary


class AttributeHeaderCSA(AttributeHeader):

    def __init_(self, fdist, vocabulary, categories):
        super(AttributeHeaderCSA, self).__init__(fdist, vocabulary, categories)

    def get_attributes(self):
        return self._categories


class FactoryRepresentation(object):

    __metaclass__ = ABCMeta

    def build(self,option):
        option = eval(option)
        return self.create(option)

    @abstractmethod
    def create(self, option):
        pass


class FactorySimpleRepresentation(FactoryRepresentation):

    def create(self, option):
        if option == EnumRepresentation.BOW:
            return FactoryBOWRepresentation()

        if option == EnumRepresentation.CSA:
            return FactoryCSARepresentation()


class AbstractFactoryRepresentation(object):

    __metaclass__ = ABCMeta

    def build_attribute_header(self, fdist, vocabulary, categories):
        return self.create_attribute_header(fdist, vocabulary, categories)

    def build_matrix_train_holder(self, space):
        return self.create_matrix_train_holder(space)

    def build_matrix_test_holder(self, space):
        return self.create_matrix_test_holder(space)

    @abstractmethod
    def create_attribute_header(self, fdist, vocabulary, categories):
        pass

    @abstractmethod
    def create_matrix_train_holder(self, space):
        pass

    @abstractmethod
    def create_matrix_test_holder(self, space):
        pass
    
    @abstractmethod
    def save_train_data(self, space):
        pass

    @abstractmethod
    def load_train_data(self, space):
        pass

class FactoryBOWRepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, categories):
        return AttributeHeaderBOW(fdist, vocabulary, categories)

    def create_matrix_train_holder(self, space):        
        self.__bow_train_matrix_holder = BOWTrainMatrixHolder(space) 
        return self.__bow_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        self.__bow_test_matrix_holder = BOWTestMatrixHolder(space) 
        return self.__bow_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__bow_train_matrix_holder is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            numpy.save(cache_file + "_mat_docs_terms.npy", 
                       self.__bow_train_matrix_holder.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.__bow_train_matrix_holder.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.__bow_train_matrix_holder.get_instance_categories())
        else:
            print "ERROR: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        self.__bow_train_matrix_holder = BOWTrainMatrixHolder(space)        
        self.__bow_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_terms.npy"))
        self.__bow_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        self.__bow_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return self.__bow_train_matrix_holder     


class FactoryCSARepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, categories):
        return AttributeHeaderCSA(fdist, vocabulary, categories)

    def create_matrix_train_holder(self, space):
        self.__csa_train_matrix_holder = CSATrainMatrixHolder(space)
        self.__csa_train_matrix_holder.build_matrix()
        return self.__csa_train_matrix_holder

    def create_matrix_test_holder(self, space):
        self.__csa_test_matrix_holder = CSATestMatrixHolder(space, self.__csa_train_matrix_holder.get_matrix_terms_concepts())
        return self.__csa_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__csa_train_matrix_holder is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            numpy.save(cache_file + "_mat_terms_concepts.npy", 
                       self.__csa_train_matrix_holder.get_matrix_terms_concepts())
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.__csa_train_matrix_holder.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.__csa_train_matrix_holder.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.__csa_train_matrix_holder.get_instance_categories())
        else:
            print "ERROR: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        self.__csa_train_matrix_holder = CSATrainMatrixHolder(space)        
        self.__csa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))        
        self.__csa_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        self.__csa_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        self.__csa_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return self.__csa_train_matrix_holder      


class MatrixHolder(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._matrix = None
        self._instance_categories = None
        self._instance_namefiles = None

    @abstractmethod
    def build_matrix(self):
        pass

    @abstractmethod
    def normalize_matrix(self, normalizer, matrix):
        pass

    @abstractmethod
    def get_matrix(self):
        pass
    
    @abstractmethod
    def get_instance_categories(self):
        pass
    
    @abstractmethod
    def get_instance_namefiles(self):
        pass

    @abstractmethod
    def set_matrix(self, value):
        pass
    
    @abstractmethod
    def set_instance_categories(self, value):
        pass
    
    @abstractmethod
    def set_instance_namefiles(self, value):
        pass

class CSAMatrixHolder(MatrixHolder):

    def __init__(self, space):
        super(CSAMatrixHolder, self).__init__()
        self.space = space

    def build_matrix_documents_concepts(self,
                                        space,
                                        virtual_classes_holder,
                                        corpus_file_list,
                                        matrix_concepts_terms):
        '''
        This function creates the self._matrix, with the following format.

            d1...d2
        a1
        .
        .
        .
        a2

        '''
        t1 = time.time()
        print "Starting CSA matrix documents-concepts"

        matrix_concepts_docs = numpy.zeros((len(space.categories), len(corpus_file_list)),
                                           dtype=numpy.float64)
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        len_vocab = len(space._vocabulary)
        unorder_dict_index = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
        ############################################################### 

        k=0
        for author in space.categories:
            archivos = virtual_classes_holder[author].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[author].dic_file_tokens[arch]
                docActualFd = nltk.FreqDist(tokens) #virtual_classes_holder[author].dic_file_fd[arch]
                tam_doc = len(tokens)

######                if len(tokens) == 0:
######                    print "***************************************************"
######                    print "TOKENS: " + str(tokens)
######                    print author
######                    print arch
######                    print "virtual_classes_holder[author].dic_file_tokens[arch]: " + str(virtual_classes_holder[author].dic_file_tokens[arch])
######                    print "Text: " + str(nltk.Text(virtual_classes_holder[author].dic_file_tokens[arch]))
######                    print "***************************************************"

                #self.weight_matrix(space, author, tokens, docActualFd, matrix_concepts_docs, 0, k, matrix_concepts_terms)
                ################################################################
                # SUPER SPEED 
                for pal in docActualFd:
                    
                    if (pal in unorder_dict_index):
                        weigh = docActualFd[pal] / float(tam_doc)
                    else:
                        weigh = 0.0
                    
                    term_vector = matrix_concepts_terms[:,unorder_dict_index[pal]] * weigh
                    matrix_concepts_docs[:,k] += term_vector
                                        
                ################################################################
                
                ################################################################
                # VERY SLOW
                
#                num_term = 0
#                for term in space._vocabulary:
#                    if term in docActualFd:
#                        weigh = (docActualFd[term]/float(tam_doc))
#                        term_vector = matrix_concepts_terms[:,num_term] * weigh
#                        matrix_concepts_docs[:,k] += term_vector
#
#                    num_term += 1
                ###############################################################

                k+=1 # contador de documentos
                
                instance_categories += [author]
                instance_namefiles += [arch]

        self._matrix = matrix_concepts_docs
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles

        t2 = time.time()
        print "End CSA matrix documents-concepts. Time: ", str(t2-t1)

    def weight_matrix(self, space, elemAutor, tokens, docActualFd, matrix_concepts_docs, numAut, k, mat_concepts_term):
        """
        La K es un contador de documentos definida en la funcion padre
        """

#        docActualFd=FreqDist(tokens)
        tamDoc=len(tokens)
        freqrank={}
        for e in docActualFd.keys():
            if str(docActualFd[e]) in freqrank.keys():
                freqrank[str(docActualFd[e])]+=1
            else:
                freqrank[str(docActualFd[e])]=1

        YULEK=0
        for e in freqrank.keys():
            YULEK+=((math.pow(float(e),2)*(float(freqrank[e])))/math.pow(float(tamDoc),2))

        YULEK*=float(10000)

        # BUG en agun momento encuentra un Docto de cero!!
        YULEK-=1/float(tamDoc+1)
        if (tamDoc == 0):
            print "tamDoc de CERO!!!: " + str(elemAutor) + " tokens: " + str(tokens)

        #print str(YULEK)

        numTermino = 0
        for pal in space._vocabulary:
            if docActualFd[pal] >= 1:
                i = numTermino
                for j in range(len(mat_concepts_term)):
                        # OJO ESTA PARTE SOLO FUNCIONA CON num_tokens_cat en los virtuales
                        cc=float(space.virtual_classes_holder_train[elemAutor].num_tokens_cat)

                        if cc==0:
                            cc=1

                        aaa=0.0
                        sss=0.0
                        
                        # OJO ESTA PARTE SOLO FUNCIONA CON num_tokens_cat en los virtuales
                        diccionarioClase = space.virtual_classes_holder_train[elemAutor].fd_vocabulary_cat

                        if pal not in diccionarioClase:
                            aaa=0.0
                            sss=0.0
                        else:
                            aaa=diccionarioClase[pal]
                            sss=diccionarioClase[pal]

                        #ANTERIOR BUENOpeso=  (docActual.diccionario[pal]/float(docActual.tamDoc))*(math.log(2+(float(aaa)/cc),2)*math.log(2+1/float(docActual.tamDoc),2))

#                        pmenos=fd[pal]
                        #for e in space.corpus.diccionarioGlobal.keys():

                        #print str(pmenos)
                        peso=  (docActualFd[pal]/float(tamDoc))#*(1.0+1.0/float(YULEK))#*(math.log(2+(float(aaa)/cc),2)/(math.log(2+1/float(tamDoc),2)))#*math.log(2+(YULEK),2)#*(diccionario[pal]/float(tamDoc))*math.log(2+(float(aaa)/cc),2)#*math.log(2+(categories[elemAutor].DiccionarioDeUso[pal] / cc1),2)
                        #print peso
                        #print space.matConceptosTerm[j,i]
                        dw=mat_concepts_term[j,i]*peso#*math.log10(1+peso)

                        matrix_concepts_docs[j+numAut,k] += dw

            numTermino += 1


class CSATrainMatrixHolder(CSAMatrixHolder):

    def __init__(self, space):
        super(CSATrainMatrixHolder, self).__init__(space)
        self._matrix_concepts_terms = None
        #self.build_matrix()

    def build_matrix_concepts_terms(self, space):
        '''
        This function creates the self._matrix_concepts_terms. The matrix
        concepts terms will has the following format:

            t_1...t_2
        a_1
        .
        .
        .
        a_n

        This determines how each author is related to each term.
        '''

        t1 = time.time()
        
        print "Starting CSA matrix concepts-terms..."

        matrix_concepts_terms = numpy.zeros((len(space.categories), len(space._vocabulary)),
                                            dtype=numpy.float64)
        
        #matrix_concepts_terms = scipy.sparse.lil_matrix(numpy.zeros((len(space.categories), len(space._vocabulary)),
        #                                                            dtype=numpy.float64))
        
        #matrix_concepts_terms = scipy.sparse.lil_matrix(numpy.zeros((len(space.categories), len(space._vocabulary)), 
        #                                                            dtype=numpy.float64))

        ################################################################
        # SUPER SPEED 
        len_vocab = len(space._vocabulary)
        unorder_dict_index = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
        ###############################################################         

        i = 0
        for author in space.categories:
            archivos = space.virtual_classes_holder_train[author].cat_file_list
            #print author
            
            total_terms_in_class = 0
            for arch in archivos:
                #print arch

                tokens = space.virtual_classes_holder_train[author].dic_file_tokens[arch]
                docActualFd = nltk.FreqDist(tokens) #space.virtual_classes_holder_train[author].dic_file_fd[arch]
                tamDoc = len(tokens)
                total_terms_in_class += tamDoc
                
                ################################################################
                # SUPER SPEED 
                
                for pal in docActualFd:
                    
                    freq = math.log((1.0 + docActualFd[pal] / float(1.0 + tamDoc)), 2) #docActualFd[pal]
                    matrix_concepts_terms[i, unorder_dict_index[pal]] += freq
                    
                
                ################################################################
                
                ################################################################
                # SUPER SLOW
                
# # # # #                 j = 0
# # # # #                 for pal in space._vocabulary:
# # # # #                     if pal in docActualFd:
# # # # #                         #print str(freq) + " antes"
# # # # # #                        freq = math.log((1 + docActualFd[pal]), 10) / math.log(1+float(tamDoc),10)
# # # # # #                        freq = math.log((1 + docActualFd[pal] / float(tamDoc)), 10) / math.log(1+float(tamDoc),10) pesado original
# # # # #         #                        freq = math.log((1 + diccionario[pal] / (2*float(tamDoc))), 2)
# # # # #                         #print pal + " : "+ str(docActualFd[pal]) + " tamDoc:" +  str(float(tamDoc))
# # # # #                         ##### PAN13: freq = math.log((1.0 + docActualFd[pal] / float(1.0 + tamDoc)), 2)
# # # # #                         freq = docActualFd[pal]
# # # # #                         ##########################################
# # # # #                         # if freq == 0.0:
# # # # #                         #     freq=0.00000001
# # # # #                         ##########################################
# # # # #                         
# # # # #                         #print str(freq) + " despues"
# # # # #                     else:
# # # # #                         freq = 0
# # # # #         #                    terminos[j] += freq
# # # # #                     matrix_concepts_terms[i,j]+=freq
# # # # # 
# # # # #                     j += 1

################################################################

            # matrix_concepts_terms[i] = matrix_concepts_terms[i]/total_terms_in_class     
            #set_printoptions(threshold='nan')
            #print matrix_concepts_terms[i]
            i+=1
            
        # PAN13 MODIFICATION -------------------------------------------
        matrix_concepts_terms = matrix_concepts_terms.transpose()
        norma=matrix_concepts_terms.sum(axis=0)
        norma+=0.00000001
        matrix_concepts_terms=matrix_concepts_terms/norma
        matrix_concepts_terms = matrix_concepts_terms.transpose()
        # --------------------------------------------------------------

        self._matrix_concepts_terms = matrix_concepts_terms

        t2 = time.time()
        print "End CSA matrix concepts-terms. Time: ", str(t2 - t1)


    def normalize_matrix(self, normalizer, matrix):
        pass

    def normalize_matrix_terms_concepts(self, matrix):
        norma=matrix.sum(axis=0) # sum all columns
        norma+=0.00000001
        matrix=matrix/norma
        return matrix
        #numpy.set_printoptions(threshold='nan')
        
        
    def __calc_entropy(self, row):
        entropy = 0
        
        for e in row:
            #print row
            if e > 0:
                entropy += (-e * math.log(e, 2))
            
        return entropy
        
    def __filter_mat_by_entropy_and_update_space_properties(self, matrix, space):
        '''
        space object is affected (throught an update space method) in this method. Moreover, this method returns
        a new matrix of terms_concepts (transposed)
        '''
        mat = matrix.transpose()
        
        debug = False       
        if debug:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            f_debug = open(cache_file + "_entropy_log.txt", "w")
        
        # In this part we perform the by entropy attribute selection -----------
        entropies_entr_index_row = []
        index=0
        for row in mat:
            the_entropy = self.__calc_entropy(row)
            entropies_entr_index_row += [(the_entropy, index, row)]
            if debug:
                f_debug.write("index: " + str(index) + " term: "+ space._vocabulary[index].encode('utf-8') + " frequency: " + str(space._fdist[space._vocabulary[index]]) + " entropy: " + str(the_entropy) + " row: " + str(row) + "\n")
            index += 1
            
        entropies_entr_index_row.sort()
        
        n_selected_terms = space.kwargs_space['select_attributes']['n_selected_terms']
        class_thresold = space.kwargs_space['select_attributes']['class_thresold']
        
        # get the n_selected_terms with low entropy, but at least in thresold classes---
        
        selected_attr = []
        no_selected_attr = []
        for c_entr, c_index, c_row in entropies_entr_index_row:
                           
            total = 0
            for e in c_row:
                if e > 0:
                    total += 1
                
            if total >= class_thresold:
                # print c_index
                selected_attr += [(c_entr, c_index)]
            else:
                no_selected_attr += [(c_entr, c_index)]
                            
                            
        entropies_entr_index = (selected_attr + no_selected_attr)[:n_selected_terms]
        
        # ----------------------------------------------------------------------
        
        # inver the elements (for easy sort()) :)
        entropies_index_entr = [ (ind, entr) for entr, ind in entropies_entr_index ]
        
        if debug:
            f_debug.write("selected(entr):" + str(entropies_index_entr) + "\n")
        
        entropies_index_entr.sort()
        
        if debug:
            f_debug.write("selected(indexados):" + str(entropies_index_entr) + "\n")
        
        new_mat_terms_concepts = numpy.zeros((n_selected_terms, len(space.categories)),
                                              dtype=numpy.float64)
        index_i = 0
        for (ind, entr) in entropies_index_entr:
            
            new_mat_terms_concepts[index_i] = mat[ind]
            
            if debug:
                f_debug.write(str(index_i) + " end_index:" + str(ind) + " term: " + space._vocabulary[ind].encode('utf-8')  + " frequency: " + str(space._fdist[space._vocabulary[ind]]) + " row: " + str(new_mat_terms_concepts[index_i]) + "\n")
            index_i += 1
        
        mat=new_mat_terms_concepts.transpose()
        
        #-----------------------------------------------------------------------
        
        # Now we need to affect the properties of our space---------------------
        
        if debug:
            f_debug.write("before_vocab: " + str(space._vocabulary) + "\n") 
        new_vocabulary = []
        for (index, entr) in entropies_index_entr:
            new_vocabulary += [space._vocabulary[index]]
        
        if debug:    
            f_debug.write("after_vocab: " + str(new_vocabulary) + "\n")                
            f_debug.write("before_fdist: " + str(space._fdist) + "\n")
            
        new_fdist = nltk.FreqDist()
        for term in new_vocabulary:
            new_fdist[term] = space._fdist[term]
        
        if debug:
            f_debug.write("after_fdist: " + str(new_fdist) + "\n")
            f_debug.write("final_vocab: " + str(new_fdist.keys()) + "\n")
            f_debug.close()
         
        # ----------------------------------------------------------------------
        
        space.update_and_save_fdist_and_vocabulary(new_fdist)
        
        return mat       

    def build_matrix(self):
        self.build_matrix_concepts_terms(self.space)
        self._matrix_concepts_terms = self.normalize_matrix_terms_concepts(self._matrix_concepts_terms)
        
        if ('select_attributes' in self.space.kwargs_space):            
            self._matrix_concepts_terms = \
            self.__filter_mat_by_entropy_and_update_space_properties(self._matrix_concepts_terms, 
                                                                     self.space)
                
        self.build_matrix_documents_concepts(self.space,
                                             self.space.virtual_classes_holder_train,
                                             self.space.corpus_file_list_train,
                                             self._matrix_concepts_terms)

    def get_matrix_terms_concepts(self):
        return self._matrix_concepts_terms

    def get_matrix(self):
        return self._matrix.transpose()
    
    def get_instance_categories(self):
        return self._instance_categories
    
    def get_instance_namefiles(self):
        return self._instance_namefiles
    
    def set_matrix_terms_concepts(self, value):
        self._matrix_concepts_terms = value
    
    def set_matrix(self, value):
        # See how we have transpose this matrix, since get method assumes it is
        # not transposed.
        self._matrix = value.transpose()
    
    def set_instance_categories(self, value):
        self._instance_categories = value
    
    def set_instance_namefiles(self, value):
        self._instance_namefiles = value


class CSATestMatrixHolder(CSAMatrixHolder):

    def __init__(self, space, matrix_concepts_terms):
        super(CSATestMatrixHolder, self).__init__(space)
        self._matrix_concepts_terms = matrix_concepts_terms
        self.build_matrix()

    def normalize_matrix(self, normalizer, matrix):
        pass

    def build_matrix(self):
        self.build_matrix_documents_concepts(self.space,
                                             self.space.virtual_classes_holder_test,
                                             self.space.corpus_file_list_test,
                                             self._matrix_concepts_terms)

    def get_matrix(self):
        return self._matrix.transpose()
    
    def get_instance_categories(self):
        return self._instance_categories
    
    def get_instance_namefiles(self):
        return self._instance_namefiles
    
    def set_matrix(self, value):
        # See how we have transpose this matrix, since get method assumes it is
        # not transposed.
        self._matrix = value.transpose()
    
    def set_instance_categories(self, value):
        self._instance_categories = value
    
    def set_instance_namefiles(self, value):
        self._instance_namefiles = value


class BOWMatrixHolder(MatrixHolder):

    def __init__(self, space):
        super(BOWMatrixHolder, self).__init__()
        self.space = space
        self.build_matrix()

    def build_matrix_doc_terminos(self,
                                  space,
                                  virtual_classes_holder,
                                  corpus_file_list):

        t1 = time.time()
        print "Starting BOW representation..."
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_terms = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_terms = numpy.zeros((len(corpus_file_list), len_vocab),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
        ###############################################################    
        
            
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = nltk.FreqDist(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                for pal in docActualFd:
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] / float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
                    if freq > 0.0:
                        rows_file.write(str(i) + "\n")
                        columns_file.write(str(unorder_dict_index[pal]) + "\n")
                        vals_file.write(str(freq) + "\n")
                    
                ################################################################

                ################################################################
                # VERY SLOW
#                j = 0
#                for pal in space._vocabulary:
#                        
#                    if (pal in docActualFd) and tamDoc > 0:
#                        #print str(freq) + " antes"
#                        freq = docActualFd[pal] / float(tamDoc) #math.log((1 + docActual.diccionario[pal] / float(docActual.tamDoc)), 10) / math.log(1+float(docActual.tamDoc),10)
##                        freq = math.log((1 + diccionario[pal] / (2*float(tamDoc))), 2)
##                        freq = math.log((1 + docActual.diccionario[pal] / (float(docActual.tamDoc))), 2)
#                        #print str(freq) + " despues"
#                        # uncomment the following line if you want a boolean weigh :)
#                        # freq=1.0
#                        #if pal == "xico":
#                        #    print pal +"where found in: "  +arch
#                    else:
#                        freq = 0
##                    terminos[j] += freq
#                    matrix_docs_terms[i,j] = freq
#
#                    j += 1
                    ############################################################

                i+=1
                
                instance_categories += [autor]
                instance_namefiles += [arch]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)

class BOWTrainMatrixHolder(BOWMatrixHolder):

    def __init__(self, space):
        super(BOWTrainMatrixHolder, self).__init__(space)

    def normalize_matrix(self):
        pass

    def build_matrix(self):
        self.build_matrix_doc_terminos(self.space,
                                       self.space.virtual_classes_holder_train,
                                       self.space.corpus_file_list_train)

    def get_matrix(self):
        return self._matrix
    
    def get_instance_categories(self):
        return self._instance_categories
    
    def get_instance_namefiles(self):
        return self._instance_namefiles
    
    def set_matrix(self, value):
        self._matrix = value
    
    def set_instance_categories(self, value):
        self._instance_categories = value
    
    def set_instance_namefiles(self, value):
        self._instance_namefiles = value


class BOWTestMatrixHolder(BOWMatrixHolder):

    def __init__(self, space):
        super(BOWTestMatrixHolder, self).__init__(space)

    def normalize_matrix(self):
        pass

    def build_matrix(self):
        self.build_matrix_doc_terminos(self.space,
                                       self.space.virtual_classes_holder_test,
                                       self.space.corpus_file_list_test)

    def get_matrix(self):
        return self._matrix
    
    def get_instance_categories(self):
        return self._instance_categories
    
    def get_instance_namefiles(self):
        return self._instance_namefiles
    
    def set_matrix(self, value):
        self._matrix = value
    
    def set_instance_categories(self, value):
        self._instance_categories = value
    
    def set_instance_namefiles(self, value):
        self._instance_namefiles = value


class LSIMatrixHolder(MatrixHolder):

    def __init__(self, space):
        super(LSIMatrixHolder, self).__init__()
        self.space = space
        self.build_matrix()

    def build_matrix_doc_terminos(self,
                                  space,
                                  virtual_classes_holder,
                                  corpus_file_list):

        t1 = time.time()
        print "Starting LSI representation..."

        matrix_docs_terms = numpy.zeros((len(corpus_file_list), len(space._vocabulary)),
                                        dtype=numpy.float64)
        i = 0

        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = nltk.FreqDist(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)

                j = 0
                for pal in space._vocabulary:
                    if pal in docActualFd:
                        #print str(freq) + " antes"
                        freq = docActualFd[pal] #math.log((1 + docActual.diccionario[pal] / float(docActual.tamDoc)), 10) / math.log(1+float(docActual.tamDoc),10)
#                        freq = math.log((1 + diccionario[pal] / (2*float(tamDoc))), 2)
#                        freq = math.log((1 + docActual.diccionario[pal] / (float(docActual.tamDoc))), 2)
                        #print str(freq) + " despues"
                        #freq=1.0
                    else:
                        freq = 0
#                    terminos[j] += freq
                    matrix_docs_terms[i,j] = freq

                    j += 1

                i+=1

        return matrix_docs_terms

        #print matConceptosTerm

        t2 = time.time()
        print Util.get_string_fancy_time(t2 - t1, "End of LSI representation.")

    def build_weight(self,
                     space,
                     virtual_classes_holder,
                     corpus_file_list):

        weighted_matrix_docs_terms = None

        #some code

        return weighted_matrix_docs_terms

    def build_lsi(self):
        final_matrix_lsi = None

        # some code

        self._matrix = final_matrix_lsi

class LSITrainMatrixHolder(LSIMatrixHolder):

    def __init__(self, space):
        super(LSITrainMatrixHolder, self).__init__(space)

    def normalize_matrix(self):
        pass

    def build_matrix(self):
        matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
                                       self.space.virtual_classes_holder_train,
                                       self.space.corpus_file_list_train)

        weighted_matrix_docs_terms = self.build_weight(self.space,
                                                       self.space.virtual_classes_holder_train,
                                                       self.space.corpus_file_list_train,
                                                       matrix_docs_terms)
        self.build_lsi(self.space,
                       self.space.virtual_classes_holder_train,
                       self.space.corpus_file_list_train,
                       weighted_matrix_docs_terms)

    def get_matrix(self):
        return self._matrix
    
    def get_instance_categories(self):
        return self._instance_categories
    
    def get_instance_namefiles(self):
        return self._instance_namefiles
    
    def set_matrix(self, value):
        self._matrix = value
    
    def set_instance_categories(self, value):
        self._instance_categories = value
    
    def set_instance_namefiles(self, value):
        self._instance_namefiles = value


class LSITestMatrixHolder(BOWMatrixHolder):

    def __init__(self, space):
        super(LSITestMatrixHolder, self).__init__(space)

    def normalize_matrix(self):
        pass

    def build_matrix(self):
        matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
                                       self.space.virtual_classes_holder_test,
                                       self.space.corpus_file_list_test)

        weighted_matrix_docs_terms = self.build_weight(self.space,
                                                       self.space.virtual_classes_holder_test,
                                                       self.space.corpus_file_list_test,
                                                       matrix_docs_terms)
        self.build_lsi(self.space,
                       self.space.virtual_classes_holder_test,
                       self.space.corpus_file_list_test,
                       weighted_matrix_docs_terms)

    def get_matrix(self):
        return self._matrix
    
    def get_instance_categories(self):
        return self._instance_categories
    
    def get_instance_namefiles(self):
        return self._instance_namefiles
    
    def set_matrix(self, value):
        self._matrix = value
    
    def set_instance_categories(self, value):
        self._instance_categories = value
    
    def set_instance_namefiles(self, value):
        self._instance_namefiles = value


class Report(object):

    def generate_report(self,
                        root,
                        experiment_name,
                        experiment_base_path,
                        global_kwargs_list):

        self.experiment_name = experiment_name
        self.experiment_base_path = experiment_base_path
        self.global_kwargs_list = global_kwargs_list
        self.root = root

        Util.create_a_dir(self.experiment_base_path)

        iterator = root.create_iterator()

        self.fancy_print_space(root, is_root=True)

        self.f_config_java_classifier = \
        open("%s/config_java_classifier_%s.txt" % (root.space_path, self.experiment_name), 'w')
        self.f_config_java_classifier.write("spaces:\n")
        self.a = 1
        self.b = 0

        for space in iterator:
            self.fancy_print_space(space)

        self.f_config_java_classifier.write("class_index: " + str(self.b))
        self.f_config_java_classifier.close()




#
#        self.root_path = ("%s/%s" % self.experiment_base_path, self.experiment_name)
#        self.root_path_sub_spaces = ("%s/sub_spaces" % self.root_path)
#
#        self.general_report(self.root)
#
#        Util.create_a_dir(self.root_path_sub_spaces)
#        i = 0
#        for sub_space in self.root.sub_spaces:
#            sub_space_path = "%s/%s_sub_space" % self.root_path_sub_spaces, str(i)
#            self.sub_spaces_report(sub_space, sub_space_path, str(i))
#
#            root_path_sssr = ("%s/%s_sssr" % sub_space_path, str(i))
#            Util.create_a_dir(root_path_sssr)
#            j = 0
#            for sssr in sub_space.sub_spaces:
#                sssr_path = "%s/%s_sssr" % root_path_sssr, str(j)
#                self.sssr_report(sssr, sssr_path,(str(i) + str(j)))
#                j += 1
#
#            i += 1

    def fancy_print_space(self, space, is_root=False):
        self.general_report(space)
        
        if is_root:
            self.create_arrfs(space)
            self.create_bin_matrices(space)
            
        self.create_properties_files(space)

        if space.is_leaf():
            self.create_details_files(space)


    def general_report(self, space):
        Util.create_a_dir(space.space_path)

        f_config = open("%s/config_%s.txt" % (space.space_path, self.experiment_name),
                        'w')

        # FIXME: Is this really necessary?, or Can you just to print the string config.?
        f_config.write("Experiment_name: %s\n" % self.experiment_name)
        f_config.write("Base_path: %s\n" % self.experiment_base_path)
        f_config.write("Space_path: %s\n" % space.space_path)
        f_config.write("Categories: %s\n" % str(space.categories))
        f_config.write("Train_corpus_config: %s\n" % str(space.train_corpus))
        f_config.write("Test_corpus_config: %s\n" % str(space.test_corpus))
        f_config.write("corpus_file_list_train: \n%s\n\n" % Util.build_fancy_list_string(space.get_train_files()))
        f_config.write("corpus_file_list_test: \n%s\n\n" % Util.build_fancy_list_string(space.get_test_files()))

        f_config.close()

    def create_arrfs(self, space):
        train_arrf_path = "%s/train_subspace%s_%s.arff" % (space.space_path, space.id_space, self.experiment_name)
        Util.write_arrf(train_arrf_path,
                        space.get_attributes(),
                        space.get_categories(),
                        space.get_matrix_train(),
                        space.get_train_files(),
                        space.get_train_instance_categories())

        test_arrf_path = "%s/test_subspace%s_%s.arff" % (space.space_path, space.id_space, self.experiment_name)
        Util.write_arrf(test_arrf_path,
                        space.get_attributes(),
                        space.get_categories(),
                        space.get_matrix_test(),
                        space.get_test_files(),
                        space.get_test_instance_categories())
        
    def create_bin_matrices(self, space):
        train_bin_path = "%s/train_subspace%s_%s" % (space.space_path, space.id_space, self.experiment_name)
        Util.write_bin_matrix(train_bin_path,
                        space.get_attributes(),
                        space.get_categories(),
                        space.get_matrix_train(),
                        space.get_train_files(),
                        space.get_train_instance_categories())

        test_bin_path = "%s/test_subspace%s_%s" % (space.space_path, space.id_space, self.experiment_name)
        Util.write_bin_matrix(test_bin_path,
                        space.get_attributes(),
                        space.get_categories(),
                        space.get_matrix_test(),
                        space.get_test_files(),
                        space.get_test_instance_categories())

    def create_properties_files(self, space):
        
        f_vocabulary_1 = open("%s/vocabulary_subspace%s_%s.txt" % (space.space_path, space.id_space, self.experiment_name), 'w')

        vocabulary_tuples_1 = Util.get_tuples_from_fdist(space.get_fdist())            

        str_vocabulary_1 = Util.build_fancy_list_string(vocabulary_tuples_1)
        #print type(str_vocabulary)
        #str_vocabulary = u'' + str_vocabulary
        #print type(str_vocabulary)
        f_vocabulary_1.write(str_vocabulary_1)
        f_vocabulary_1.close()
        
        

        f_vocabulary = open("%s/vocabulary_subspace_Simple%s_%s.txt" % (space.space_path, space.id_space, self.experiment_name), 'w')

        vocabulary_tuples_temp = Util.get_tuples_from_fdist(space.get_fdist())        
        
        vocabulary_tuples = []
        for v_tuple in vocabulary_tuples_temp:
            element = v_tuple[0]
            number = v_tuple[1]
            # PAN13: print "VOCABULARY ELEMENTS: "
            # PAN13: print element.encode('utf-8')
            # PAN13: print type(element)
            vocabulary_tuples +=[(element.encode('utf-8'), number)]
            

        str_vocabulary = Util.build_fancy_vocabulary(vocabulary_tuples)
        #print type(str_vocabulary)
        #str_vocabulary = u'' + str_vocabulary
        #print type(str_vocabulary)
        f_vocabulary.write(str_vocabulary)
        f_vocabulary.close()
        
        # FIXME: These lines have not been debugged.
        json_vocabulary = open("%s/vocabulary_subspace_Json%s_%s.json" % (space.space_path, space.id_space, self.experiment_name), 'w')
        json.dump(space.get_fdist().keys(), json_vocabulary)
        json_vocabulary.close()
        # ----------------------------------------------------------------------

    def create_details_files(self, space):
        self.create_details(space.space_path,
                            space.id_space,
                            self.experiment_name,
                            space.get_virtual_classes_holder_train(),
                            "train")

        self.create_details(space.space_path,
                            space.id_space,
                            self.experiment_name,
                            space.get_virtual_classes_holder_test(),
                            "test")


        self.b += len(space.get_attributes())

        self.f_config_java_classifier.write("- {a: " + str(self.a) + ", " + "b: " + str(self.b) + "}\n")
        #self.f_config_java_classifier.write()

        self.a += len(space.get_attributes())


    def create_details(self, space_path, id_space, experiment_name, virtual_classes_holder, which_dataset):
        f_details = open("%s/details_subspace%s_%s_%s.txt" % (space_path, id_space, experiment_name, which_dataset), 'w')

        virtual_categories = virtual_classes_holder

        for key in virtual_categories.keys():
            virtual_category = virtual_categories[key]
            f_details.write("Author: %s\n" % virtual_category.author)
            # OJO SOLO FUNCIONA CON LOS VIRTUALES FULL
            # f_details.write("Number of tokens: %s\n" % virtual_category.num_tokens_cat)
            # f_details.write("Author Vocabulary: \n%s\n" % Util.build_fancy_list_string(Util.get_tuples_from_fdist(virtual_category.fd_vocabulary_cat)))
            f_details.write("File's list:\n%s\n" % Util.build_fancy_list_string(virtual_category.cat_file_list))

            # THE NEXT LINES ONLY WORKS USING FULL VIRTUALS
            #===================================================================
            # Build the vocabulary tuples for each file
            #===================================================================
            # vocabulary_file_tuples = Util.get_tuples_from_fdist(virtual_category.dic_file_fd)

            # str_vocabulary_file = Util.build_fancy_list_string(vocabulary_file_tuples)

            # f_details.write("Vocabulary per file:\n" + str_vocabulary_file + "\n")
            #===================================================================

        f_details.close()




        #path, attributes, categories, matrix, name_files, relation_name="my_relation"

#    def space_report(self, sub_space, sub_space_path, str_n):
#        Util.create_a_dir(sub_space_path)
#
#        vocabulary_string = Util.build_fancy_list_string(self.root, 3)
#        vocabulary_sub_space_path = "%s/%s_vocabulary_%s", sub_space_path, str_n, self.experiment_name
#
#        f_vocabulary_sub_space = open(vocabulary_sub_space_path)
#        f_vocabulary_sub_space.write(vocabulary_string)
#        f_vocabulary_sub_space.close()
#
#        train_arrf_path = "%s/%s_train_arrf_%s" % sub_space_path, str_n, self.experiment_name
#        Util.write_arrf(train_arrf_path,
#                        sub_space.get_attributes(),
#                        sub_space.get_categories(),
#                        sub_space.get_matrix_train(),
#                        sub_space.get_train_files())
#
#        test_arrf_path = "%s/%s_test_arrf_%s" % sub_space_path, str_n, self.experiment_name
#        Util.write_arrf(test_arrf_path,
#                        sub_space.get_attributes(),
#                        sub_space.get_categories(),
#                        sub_space.get_matrix_test(),
#                        sub_space.get_test_files())

class ReportPart(object):
    
    def generate_report_part(self,
                        root,
                        experiment_name,
                        experiment_base_path,
                        global_kwargs_list,
                        part_train=True):

        self.experiment_name = experiment_name
        self.experiment_base_path = experiment_base_path
        self.global_kwargs_list = global_kwargs_list
        self.root = root
        self.part_train = part_train

        Util.create_a_dir(self.experiment_base_path)

        iterator = root.create_iterator()

        self.fancy_print_space_part(root, is_root=True)

        self.f_config_java_classifier = \
        open("%s/%s_config_java_classifier_%s.txt" % (root.space_path, str(("Test", "Train")[self.part_train]), self.experiment_name), 'w')
        self.f_config_java_classifier.write("spaces:\n")
        self.a = 1
        self.b = 0

        for space in iterator:
            self.fancy_print_space_part(space)

        self.f_config_java_classifier.write("class_index: " + str(self.b))
        self.f_config_java_classifier.close()
        
    def fancy_print_space_part(self, space, is_root=False):
        self.general_report_part(space)
        
        if is_root:
            self.create_arrfs_part(space)
            self.create_bin_matrices_part(space)
            
        self.create_properties_files_part(space)

        if space.is_leaf():
            self.create_details_files_part(space)


    def general_report_part(self, space):
        Util.create_a_dir(space.space_path)

        f_config = open("%s/%s_config_%s.txt" % (space.space_path, str(("Test", "Train")[self.part_train]), self.experiment_name),
                        'w')

        # FIXME: Is this really necessary?, or Can you just to print the string config.?
        f_config.write("Experiment_name: %s\n" % self.experiment_name)
        f_config.write("Base_path: %s\n" % self.experiment_base_path)
        f_config.write("Space_path: %s\n" % space.space_path)
        f_config.write("Categories: %s\n" % str(space.categories))
        
        if self.part_train:
            f_config.write("Train_corpus_config: %s\n" % str(space.train_corpus))
            f_config.write("corpus_file_list_train: \n%s\n\n" % Util.build_fancy_list_string(space.get_train_files()))
        else:
            f_config.write("Test_corpus_config: %s\n" % str(space.test_corpus))
            f_config.write("corpus_file_list_test: \n%s\n\n" % Util.build_fancy_list_string(space.get_test_files()))

        f_config.close()

    def create_arrfs_part(self, space):
        
        if self.part_train:            
            train_arrf_path = "%s/train_subspace%s_%s.arff" % (space.space_path, space.id_space, self.experiment_name)
            Util.write_arrf(train_arrf_path,
                            space.get_attributes(),
                            space.get_categories(),
                            space.get_matrix_train(),
                            space.get_train_files(),
                            space.get_train_instance_categories())
        else:
            test_arrf_path = "%s/test_subspace%s_%s.arff" % (space.space_path, space.id_space, self.experiment_name)
            Util.write_arrf(test_arrf_path,
                            space.get_attributes(),
                            space.get_categories(),
                            space.get_matrix_test(),
                            space.get_test_files(),
                            space.get_test_instance_categories())
        
    def create_bin_matrices_part(self, space):
        
        if self.part_train:
             
            train_bin_path = "%s/train_subspace%s_%s" % (space.space_path, space.id_space, self.experiment_name)
            Util.write_bin_matrix(train_bin_path,
                            space.get_attributes(),
                            space.get_categories(),
                            space.get_matrix_train(),
                            space.get_train_files(),
                            space.get_train_instance_categories())
            
            train_bin_path_extra = "%s/train_subspace%s_%s" % (space.space_path, space.id_space, self.experiment_name)
            numpy.save(train_bin_path_extra + "_instance_categories.npy", numpy.array(space.get_train_instance_categories()))
            numpy.save(train_bin_path_extra + "_instance_namefiles.npy", numpy.array(space.get_train_files()))
        else:
            
            test_bin_path = "%s/test_subspace%s_%s" % (space.space_path, space.id_space, self.experiment_name)
            Util.write_bin_matrix(test_bin_path,
                            space.get_attributes(),
                            space.get_categories(),
                            space.get_matrix_test(),
                            space.get_test_files(),
                            space.get_test_instance_categories())
            
            test_bin_path_extra = "%s/test_subspace%s_%s" % (space.space_path, space.id_space, self.experiment_name)
            numpy.save(test_bin_path_extra + "_instance_categories.npy", numpy.array(space.get_test_instance_categories()))
            numpy.save(test_bin_path_extra + "_instance_namefiles.npy", numpy.array(space.get_test_files()))

    def create_properties_files_part(self, space):
        
        if self.part_train:
            f_vocabulary_1 = open("%s/%s_vocabulary_subspace%s_%s.txt" % (space.space_path, str(("Test", "Train")[self.part_train]), space.id_space, self.experiment_name), 'w')
        
            vocabulary_tuples_1 = Util.get_tuples_from_fdist(space.get_fdist())            
        
            str_vocabulary_1 = Util.build_fancy_list_string(vocabulary_tuples_1)
            #print type(str_vocabulary)
            #str_vocabulary = u'' + str_vocabulary
            #print type(str_vocabulary)
            f_vocabulary_1.write(str_vocabulary_1)
            f_vocabulary_1.close()
            
            
            f_vocabulary = open("%s/%s_vocabulary_subspace_Simple%s_%s.txt" % (space.space_path, str(("Test", "Train")[self.part_train]), space.id_space, self.experiment_name), 'w')
        
            vocabulary_tuples_temp = Util.get_tuples_from_fdist(space.get_fdist())        
            
            vocabulary_tuples = []
            for v_tuple in vocabulary_tuples_temp:
                element = v_tuple[0]
                number = v_tuple[1]
                # PAN13: print "VOCABULARY ELEMENTS: "
                # PAN13: print element.encode('utf-8')
                # PAN13: print type(element)
                vocabulary_tuples +=[(element.encode('utf-8'), number)]
                
        
            str_vocabulary = Util.build_fancy_vocabulary(vocabulary_tuples)
            #print type(str_vocabulary)
            #str_vocabulary = u'' + str_vocabulary
            #print type(str_vocabulary)
            f_vocabulary.write(str_vocabulary)
            f_vocabulary.close()
            
            json_vocabulary =  open("%s/%s_vocabulary_subspace_json_%s_%s.json" % (space.space_path, str(("Test", "Train")[self.part_train]), space.id_space, self.experiment_name), 'w')
            json.dump(space.get_fdist().keys(), json_vocabulary)
            json_vocabulary.close()
        else:
            pass

    def create_details_files_part(self, space):
        
        if self.part_train: 
                
            self.create_details_part(space.space_path,
                                space.id_space,
                                self.experiment_name,
                                space.get_virtual_classes_holder_train(),
                                "train")
            
        else:

            self.create_details_part(space.space_path,
                                space.id_space,
                                self.experiment_name,
                                space.get_virtual_classes_holder_test(),
                                "test")


        self.b += len(space.get_attributes())

        self.f_config_java_classifier.write("- {a: " + str(self.a) + ", " + "b: " + str(self.b) + "}\n")
        #self.f_config_java_classifier.write()

        self.a += len(space.get_attributes())


    def create_details_part(self, space_path, id_space, experiment_name, virtual_classes_holder, which_dataset):
        f_details = open("%s/%s_details_subspace%s_%s_%s.txt" % (space_path, str(("Test", "Train")[self.part_train]), id_space, experiment_name, which_dataset), 'w')

        virtual_categories = virtual_classes_holder

        for key in virtual_categories.keys():
            virtual_category = virtual_categories[key]
            f_details.write("Author: %s\n" % virtual_category.author)
            # OJO SOLO FUNCIONA CON LOS VIRTUALES FULL
            # f_details.write("Number of tokens: %s\n" % virtual_category.num_tokens_cat)
            # f_details.write("Author Vocabulary: \n%s\n" % Util.build_fancy_list_string(Util.get_tuples_from_fdist(virtual_category.fd_vocabulary_cat)))
            f_details.write("File's list:\n%s\n" % Util.build_fancy_list_string(virtual_category.cat_file_list))

            # THE NEXT LINES ONLY WORKS USING FULL VIRTUALS
            #===================================================================
            # Build the vocabulary tuples for each file
            #===================================================================
            # vocabulary_file_tuples = Util.get_tuples_from_fdist(virtual_category.dic_file_fd)

            # str_vocabulary_file = Util.build_fancy_list_string(vocabulary_file_tuples)

            # f_details.write("Vocabulary per file:\n" + str_vocabulary_file + "\n")
            #===================================================================

        f_details.close()
        

class Corpora(object):
    '''
    This class transforms all the files ".dat" into a readable ".txt" files,
    using the respective structure for the corpus.

    Note that this class HIGHLY depends on the structure of the name for the
    ".dat" files, which MUST to have the following form:
    <id number>_<term name with a camel case convention>_<corpus partition name with a camel case convention>.dat

    some examples are as follows: "1_TermRegExp_test.dat", "2_TokenLenght_train.dat", etc.

    The camel case convention is important in order to build the appropiate
    directories through the regular expressions defined in this class.
    '''

    def __init__(self, path_of_sources):
        self.path_of_sources = path_of_sources

    def generate(self):
        Util.create_a_dir('%s/corpora' % (self.path_of_sources))

        term_corpus_files = glob.glob('%s/%s' % (self.path_of_sources, '*.dat'))

        for term_corpus_file in term_corpus_files:
            # print term_corpus_file

            # ------------------------------------------------------------------
            # Extract the name of the file, in order to use the same name for
            # for its dir.
            # ------------------------------------------------------------------

            match = re.match('.+/(\d+_.+)_.+.dat', term_corpus_file)
            corporita_name = match.group(1)
            print corporita_name

            path_corporita = '%s/corpora/corporita_%s' % (self.path_of_sources,
                                                          corporita_name)

            Util.create_a_dir(path_corporita)

            match = re.match('.+/(\d+_.+).dat', term_corpus_file)
            corpus_name = match.group(1)
            print corpus_name

            path_corpus = '%s/corpus_%s' % (path_corporita,
                                            corpus_name)

            Util.create_a_dir(path_corpus)
            # ------------------------------------------------------------------

#            match = re.match('.+/\d+_(.*)_.*\.dat', term_corpus_file)
#            name_corpus = match.group(1)
#
#            match = re.match('.+/\d+_.*_(.*)\.dat', term_corpus_file)
#            type_corpus = match.group(1)

            shelf = shelve.open(term_corpus_file, protocol=2)

            for f in sorted(shelf.keys()):

                match = re.match('(.+)/(.+)', f)
                # uncomment if you want to see the name of the corpus
                # print match.group(1)

                path_temp = '%s/%s' % (path_corpus,
                                       match.group(1))

                Util.create_a_dir(path_temp)

                # print '%s/%s' % (path_temp, match.group(2))

                temp_f = open('%s/%s' % (path_temp, match.group(2)), 'w')
                contents = u' '.join(shelf[f])
                temp_f.write(contents.encode("utf-8"))
                temp_f.close()

            shelf.close()


class SpaceComponent(object):

    __metaclass__ = ABCMeta

    def __init__(self,
                 id_space,
                 space_path,
                 categories,
                 train_corpus,
                 test_corpus,
                 corpus_file_list_train,
                 corpus_file_list_test,
                 kwargs_space,
                 processing_option,
                 root):

        self.id_space = id_space
        self.space_path = space_path
        self.categories = categories
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self.corpus_file_list_train = corpus_file_list_train
        self.corpus_file_list_test = corpus_file_list_test
        self.root = root
        self.kwargs_space = kwargs_space

        self.virtual_elements = None
        self.tokens = None
        self._fdist = None
        self._vocabulary = None
        self._attributes = None
        self._representation = None

        self._matrix_train = None
        self._matrix_test = None

        self.virtual_classes_holder_train = None
        self.virtual_classes_holder_test = None

        self.processing_option = processing_option

        factory_simple_terms_processing = virtuals.FactorySimpleTermsProcessing()
        
        # FIXME: factory_processing is an Abstract_Factory, maybe we have to
        # consider changing the name of the class in order to reflect this.
        # For instance, FactoryVirtuals.
        self.factory_processing = factory_simple_terms_processing.build(self.processing_option)

    def set_virtual_elements(self, value):
        self.virtual_elements = value

    def get_virtual_classes_holder_train(self):
        return self.virtual_classes_holder_train

    def get_virtual_classes_holder_test(self):
        return self.virtual_classes_holder_test
    
    def update_and_save_fdist_and_vocabulary(self, new_fdist):
        
        self._fdist = new_fdist
        self._vocabulary = self._fdist.keys() 
        
        Util.create_a_dir(self.space_path)
        
        cache_file = "%s/%s_properties.dat" % (self.space_path, self.id_space)
        shelf = shelve.open(cache_file, protocol=1)
        
        shelf[cache_file + "_vocabulary"] = self._vocabulary
        shelf[cache_file + "_fdist"] = self._fdist
        shelf.close()
        
        for vc_train in self.virtual_classes_holder_train :
            self.virtual_classes_holder_train[vc_train].dic_file_tokens.set_fdist(self._fdist)
        
        for vc_test in self.virtual_classes_holder_test: 
            self.virtual_classes_holder_test[vc_test].dic_file_tokens.set_fdist(self._fdist)
        
    
    def create_space_properties_and_save(self):
        
        self.create_space_properties()
        
        Util.create_a_dir(self.space_path)
        
        cache_file = "%s/%s_properties.dat" % (self.space_path, self.id_space)
        shelf = shelve.open(cache_file, protocol=1)
        
        shelf[cache_file + "_vocabulary"] = self._vocabulary
        shelf[cache_file + "_fdist"] = self._fdist
        shelf.close()
        
        # save unecesary info for serialization --------------------------------
        temp_corpuses = []
        for kwargs_space_term in self.kwargs_space['terms']:
            temp_corpuses += [kwargs_space_term['corpus']]
            kwargs_space_term['corpus'] = "NOSERIALIZED"
            
        temp_sources = []
        for kwargs_space_term in self.kwargs_space['terms']:
            temp_sources += [[kwargs_space_term['source']]]
            kwargs_space_term['source'] = "NOSERIALIZED" 
        
        # ----------------------------------------------------------------------
            
        yaml.dump(self.kwargs_space, open(cache_file + "_kwargs_space.yaml", 'w'))
        
        # restore unecesary info for serialization --------------------------------
        
        for (kwargs_space_term, temp_corpus) in zip(self.kwargs_space['terms'], temp_corpuses):
            kwargs_space_term['corpus'] = temp_corpus                    
        
        for (kwargs_space_term, temp_source) in zip(self.kwargs_space['terms'], temp_sources):
            kwargs_space_term['source'] = temp_source 
        
        # ----------------------------------------------------------------------
        
        
    def load_space_properties_(self):
        
        cache_file = "%s/%s_properties.dat" % (self.space_path, self.id_space)
        
        shelf = shelve.open(cache_file, protocol=1)
        
        self._vocabulary = shelf[cache_file + "_vocabulary"] 
        self._fdist = shelf[cache_file + "_fdist"]
            
        self.kwargs_space = yaml.load(open(cache_file + "_kwargs_space.yaml", 'r'))    
        
        shelf.close()
        

    def create_space_properties(self):
        
        # ==================================================================
        # In the following lines we just use self.kwargs_space['terms'], cuz
        # they contain the especifications (obvioulsly  it will extract all from
        # the TRAINING CORPUS). This is because this is the corpus on which in 
        # classification task we are just allow to see. 
        # ==================================================================
        
        # VirtualElements (VirtualTerms or VirtualVocabulary) stores the
        # state (fdist, vocabulary,...and kwargs_specific that tell us
        # how the Vocabulary or Terms wered filtered.

        if (self.root == None):
            # ==================================================================
            # If this is the root, build the virtual_elements and apply the
            # global filters to get the tokens.
            # ==================================================================
            print "STEP: 1 ..."
            virtual_processor = \
            self.factory_processing.\
            build_virtual_processor(self.kwargs_space['terms'])
            
            self.set_virtual_elements(virtual_processor.virtual_elements)
            
            print "STEP: 2 ..."

            virtual_global_processor = \
            self.factory_processing.\
            build_virtual_global_processor(self.virtual_elements,
                                           self.kwargs_space['filters_terms'])

            self._fdist = virtual_global_processor.fdist
            
            print "END OF STEPS 1 AND 2..."
#
#            FilterTermsVirtualGlobalProcessor(self.virtual_elements,
#                                              self.kwargs_space['filters_terms'])

            #build_virtual_global_processor(self.virtual_elements,
            #                               self.kwargs_space['filters_terms'])



            # ==================================================================
        else:
            # This is not the root so ...
            self.virtual_elements = []

            if ('ids_term' in self.kwargs_space['terms']):
                # ==============================================================
                # This is not the root, but it gives just the ids, which means
                # that inherit the exact selected terms that its root filtered.
                # ==============================================================

                virtual_terms = \
                [virtual_element
                 for virtual_element in self.get_virtual_elements_root()
                 if virtual_element.id_term in self.kwargs_space['terms']['ids_term']]

                self.virtual_elements = virtual_terms

                # finally we apply the global filter of this particular self

                virtual_global_processor = \
                FilterTermsVirtualGlobalProcessor(self.virtual_elements,
                                             self.kwargs_space['filters_terms'])

                # ==============================================================

            else:

                # VirtualElements (VirtualTerms or VirtualVocabulary) stores the
                # state (fdist, vocabulary,...and kwargs_specific that tell us
                # how the Vocabulary or Terms wered filtered.
                 
                # ==============================================================
                # This is not the root,CreateSpace but it gives specific definition of how
                # the terms will be filtered.
                # ==============================================================
                i = 0
                for kwargs_term in self.kwargs_space['terms']:

                    # POINT_1 --------------------------------------------------
                    # Save the filter of the YAML in this node, because it could be
                    # true that we want terms in the parent node, BUT we don't want
                    # the same way to filter !!!!
                    # ----------------------------------------------------------
                    
                    # 1.- (ALT_EXPLANATION) : ARGS LEVEL ...  Get(Save) the filters of this space
                    if 'filters_terms' in kwargs_term:
                        filters_terms = kwargs_term['filters_terms']
                        
                        
                    # 2.- (ALT_EXPLANATION) : OBJECT VIRTUAL LEVEL: Check all virtuals of the parent 
                    # and reatin those with the same id in this space (1***).
                    
                    for virtual_element in self.get_virtual_elements_root():
                        #match = re.match('.+/([0-9]+)', virtual_element.id_term)
                        
                        # 3.- (ALT_EXPLANATION) : (1***) OBJECT VIRTUAL LEVEL
                        if (virtual_element.id_term == kwargs_term['id_term']):
                            self.kwargs_space['terms'][i] = virtual_element.kwargs_term

                            # POINT_2 ------------------------------------------
                            # Filter in a specific way (per very term)
                            # This part OVERRIDES filters of the parent. Check
                            # how in the above line a identical copy of the
                            # terms specification is made on this node:
                            # "self.kwargs_space['terms'][i] = virtual_element.kwargs_term"
                            # However we don't want to filter vocabulary in the same way as
                            # the parent did. So, we override the value loaded in the
                            # POINT_1.
                            
                            # 4.- (ALT_EXPLANATION) : ARGS LEVEL.... put the child filters to the parent node :)
                            # to the arguments.
                            if 'filters_terms' in kwargs_term:
                                self.kwargs_space['terms'][i]['filters_terms'] = filters_terms
                                
                                
                            # 5.- (ALT_EXPLANATION) : OBJECT VIRTUAL LEVEL Here we retain the identical virtuals of the parent
                            self.virtual_elements += [virtual_element]

                    i += 1

                # 6.- (ALT_EXPLANATION): REFILTER THE VIRTUAL OBJECT USING ARGS LEVEL :) (IN PAIRS)
                virtual_re_processor = \
                self.factory_processing.\
                build_virtual_re_processor(self.virtual_elements,
                                         self.kwargs_space['terms'])

#                FilterTermsVirtualReProcessor(self.virtual_elements,
#                                              self.kwargs_space['terms'])

                self.virtual_elements = \
                virtual_re_processor.new_virtual_elements

                # APPLY THE GLOBAL FILTERS
                virtual_global_processor = \
                self.factory_processing.\
                build_virtual_global_processor(self.virtual_elements,
                                               self.kwargs_space['filters_terms'])

                self._fdist = virtual_global_processor.fdist

#                FilterTermsVirtualGlobalProcessor(self.virtual_elements,
#                                             self.kwargs_space['filters_terms'])


                # ==============================================================
        if (self.processing_option == virtuals.EnumTermsProcessing.FULL):
            self.tokens = virtual_global_processor.tokens

        self._vocabulary = self._fdist.keys()
        #self._fdist = nltk.FreqDist(self.tokens)FULL
        #self._vocabulary = self._fdist.keys()
        #print self._vocabulary


    @abstractmethod
    def is_leaf(self):
        pass

    @abstractmethod
    def create_virtuals(self):
        pass

    @abstractmethod
    def create_representation(self):
        pass
    
    @abstractmethod
    def create_train_representation_and_save(self):
        pass
    
    @abstractmethod
    def load_train_and_create_test_representation(self):
        pass

    @abstractmethod
    def get_attributes(self):
        pass

    @abstractmethod
    def get_categories(self):
        pass

    @abstractmethod
    def get_matrix_train(self):
        pass

    @abstractmethod
    def get_matrix_test(self):
        pass

    @abstractmethod
    def get_train_files(self):
        pass

    @abstractmethod
    def get_test_files(self):
        pass
    
    @abstractmethod
    def get_train_instance_categories(self):
        pass

    @abstractmethod
    def get_test_instance_categories(self):
        pass

    def get_vocabulary(self):
        return self.vocabulary

    def set_vocabulary(self, value):
        self._vocabulary = value

    def get_fdist(self):
        return self._fdist

    def set_fdist(self, value):
        self._fdist = value

    @abstractmethod
    def get_tokens(self):
        pass

    @abstractmethod
    def add(self, space_component):
        pass

    @abstractmethod
    def remove(self, space_component):
        pass

    @abstractmethod
    def get_child(self, i):
        pass

    @abstractmethod
    def create_iterator(self):
        pass

    def get_virtual_elements_root(self):
        return self.root.virtual_elements


class SpaceComposite(SpaceComponent):

    def __init__(self,
                 id_space,
                 space_path,
                 categories,
                 train_corpus,
                 test_corpus,
                 corpus_file_list_train,
                 corpus_file_list_test,
                 kwargs_space,
                 processing_option,
                 root):

        super(SpaceComposite, self).__init__(id_space,
                                             space_path,
                                             categories,
                                             train_corpus,
                                             test_corpus,
                                             corpus_file_list_train,
                                             corpus_file_list_test,
                                             kwargs_space,
                                             processing_option,
                                             root)

        self.space_components = []

    def is_leaf(self):
        return False

    def create_virtuals(self):
        raise UnsupportedOperationError("Unsupported operation")

    def create_representation(self):
        raise UnsupportedOperationError("Unsupported operation")
    
    def create_train_representation_and_save(self):
        raise UnsupportedOperationError("Unsupported operation")
    
    def load_train_and_create_test_representation(self):
        raise UnsupportedOperationError("Unsupported operation")

    def get_attributes(self):
        attributes = []

        for space_component in self.space_components:
            attributes += space_component.get_attributes()

        return attributes

    def get_categories(self):
        return self.categories

    def get_matrix_train(self):
        matrix_train = None

        for space_component in self.space_components:
            if (matrix_train == None):
                matrix_train = space_component.get_matrix_train()
            else:
                matrix_train = \
                numpy.c_[matrix_train, space_component.get_matrix_train()]

        return matrix_train

    def get_matrix_test(self):
        matrix_test = None

        for space_component in self.space_components:
            if (matrix_test == None):
                matrix_test = space_component.get_matrix_test()
            else:
                matrix_test = \
                numpy.c_[matrix_test, space_component.get_matrix_test()]

        return matrix_test

    def get_train_files(self):
        train_instance_namefiles = None

        for space_component in self.space_components:
            if (train_instance_namefiles == None):
                train_instance_namefiles = space_component.get_train_files()
            else:
                
                flag_identical = True
                
                if len(train_instance_namefiles) != len(space_component.get_train_files()):
                    flag_identical = False
                    print "The Size is different!!!: ", len(train_instance_namefiles), " VS ", len(space_component.get_train_files())  
                    
                for a, b in zip (train_instance_namefiles, space_component.get_train_files()):
                    if a != b:
                        flag_identical = False
                        print "Elements are not the same!!!: ", a, " VS ", b
                        break
                    
                if flag_identical:
                    train_instance_namefiles = space_component.get_train_files()
                else:
                    raise NonIdenticalInstancesOfSubspacesError\
                        ("The subspaces file lists are non identical!!!:" +
                         " please check the length or elements.")

        return train_instance_namefiles
        
        #return self.corpus_file_list_train

    def get_test_files(self):
        test_instance_namefiles = None

        for space_component in self.space_components:
            if (test_instance_namefiles == None):
                test_instance_namefiles = space_component.get_test_files()
            else:
                
                flag_identical = True
                
                if len(test_instance_namefiles) != len(space_component.get_test_files()):
                    flag_identical = False
                    print "The Size is different!!!: ", len(test_instance_namefiles), " VS ", len(space_component.get_test_files())
                    
                for a, b in zip (test_instance_namefiles, space_component.get_test_files()):
                    if a != b:
                        flag_identical = False
                        print "Elements are not the same!!!: ", a, " VS ", b
                        break
                    
                if flag_identical:
                    test_instance_namefiles = space_component.get_test_files()
                else:
                    raise NonIdenticalInstancesOfSubspacesError\
                        ("The subspaces file lists are non identical!!!:" +
                         " please check the length or elements.")

        return test_instance_namefiles
        
        #return self.corpus_file_list_test
        
    def get_train_instance_categories(self):
        train_instance_categories = None

        for space_component in self.space_components:
            if (train_instance_categories == None):
                train_instance_categories = space_component.get_train_instance_categories()
            else:
                
                flag_identical = True
                
                if len(train_instance_categories) != len(space_component.get_train_instance_categories()):
                    flag_identical = False
                    print "The Size is different (INSTANCE_CAT)!!!: ", len(train_instance_categories), " VS ", len(space_component.get_train_instance_categories())  
                    
                for a, b in zip (train_instance_categories, space_component.get_train_instance_categories()):
                    if a != b:
                        flag_identical = False
                        print "Elements are not the same (INSTANCE_CAT)!!!: ", a, " VS ", b
                        break
                    
                if flag_identical:
                    train_instance_categories = space_component.get_train_instance_categories()
                else:
                    raise NonIdenticalInstancesOfSubspacesError\
                        ("The subspaces file lists are non identical (INSTANCE_CAT)!!!:" +
                         " please check the length or elements.")

        return train_instance_categories
        
        #return self.corpus_file_list_train

    def get_test_instance_categories(self):
        test_instance_categories = None

        for space_component in self.space_components:
            if (test_instance_categories == None):
                test_instance_categories = space_component.get_test_instance_categories()
            else:
                
                flag_identical = True
                
                if len(test_instance_categories) != len(space_component.get_test_instance_categories()):
                    flag_identical = False
                    print "The Size is different (INSTANCE_CAT)!!!: ", len(test_instance_categories), " VS ", len(space_component.get_test_instance_categories())
                    
                for a, b in zip (test_instance_categories, space_component.get_test_instance_categories()):
                    if a != b:
                        flag_identical = False
                        print "Elements are not the same (INSTANCE_CAT)!!!: ", a, " VS ", b
                        break
                    
                if flag_identical:
                    test_instance_categories = space_component.get_test_instance_categories()
                else:
                    raise NonIdenticalInstancesOfSubspacesError\
                        ("The subspaces file lists are non identical (INSTANCE_CAT)!!!:" +
                         " please check the length or elements.")

        return test_instance_categories
        
        #return self.corpus_file_list_test

    def get_tokens(self):
        pass

    def add(self, space_component):
        self.space_components.append(space_component)

    def remove(self, space_component):
        self.space_components.remove(space_component)

    def get_child(self, i):
        return self.space_components[i]

    def create_iterator(self):
        return SpaceCompositeIterator(iter(self.space_components))


class SpaceItem(SpaceComponent):

    def __init__(self,
                 id_space,
                 space_path,
                 categories,
                 train_corpus,
                 test_corpus,
                 corpus_file_list_train,
                 corpus_file_list_test,
                 kwargs_space,
                 processing_option,
                 root):

        super(SpaceItem, self).__init__(id_space,
                                        space_path,
                                        categories,
                                        train_corpus,
                                        test_corpus,
                                        corpus_file_list_train,
                                        corpus_file_list_test,
                                        kwargs_space,
                                        processing_option,
                                        root)

    def create_virtuals(self):
        self.virtual_classes_holder_train = \
        VirtualCategoriesHolder(self.categories,
                                self.train_corpus,
                                self.kwargs_space['terms'],
                                self._fdist,
                                self.corpus_file_list_train).virtual_categories

        self.virtual_classes_holder_test = \
        VirtualCategoriesHolder(self.categories,
                                self.test_corpus,
                                self.kwargs_space['terms'],
                                self._fdist,
                                self.corpus_file_list_test).virtual_categories

        print "Virtuals has been built"


    def create_train_representation_and_save(self):
        self.factory_representation = FactorySimpleRepresentation()

        self.representation = \
        self.factory_representation.build(self.kwargs_space['representation'])

        self.attribute_header = \
        self.representation.build_attribute_header(self._fdist,
                                                   self._vocabulary,
                                                   self.categories)

####        print "DATA ######"
        print self.kwargs_space['representation']
####        print self.attribute_header
####        print self.attribute_header.get_attributes()
####        print self.attribute_header._fdist
####        print self.attribute_header._vocabulary
####        print self.attribute_header._categories
####        print "DATA ######"

        self.matrix_train_holder = \
        self.representation.build_matrix_train_holder(self)
        
        self.representation.save_train_data(self)

        # self.matrix_test_holder = \
        # self.representation.build_matrix_test_holder(self)                
        
    def load_train_and_create_test_representation(self):
        self.factory_representation = FactorySimpleRepresentation()

        self.representation = \
        self.factory_representation.build(self.kwargs_space['representation'])

        self.attribute_header = \
        self.representation.build_attribute_header(self._fdist,
                                                   self._vocabulary,
                                                   self.categories)

####        print "DATA ######"
        print self.kwargs_space['representation']
####        print self.attribute_header
####        print self.attribute_header.get_attributes()
####        print self.attribute_header._fdist
####        print self.attribute_header._vocabulary
####        print self.attribute_header._categories
####        print "DATA ######"

        self.matrix_train_holder = \
        self.representation.load_train_data(self)

        self.matrix_test_holder = \
        self.representation.build_matrix_test_holder(self)
        
        
    def create_representation(self):
        self.factory_representation = FactorySimpleRepresentation()

        self.representation = \
        self.factory_representation.build(self.kwargs_space['representation'])

        self.attribute_header = \
        self.representation.build_attribute_header(self._fdist,
                                                   self._vocabulary,
                                                   self.categories)

####        print "DATA ######"
        print self.kwargs_space['representation']
####        print self.attribute_header
####        print self.attribute_header.get_attributes()
####        print self.attribute_header._fdist
####        print self.attribute_header._vocabulary
####        print self.attribute_header._categories
####        print "DATA ######"

        self.matrix_train_holder = \
        self.representation.build_matrix_train_holder(self)

        self.matrix_test_holder = \
        self.representation.build_matrix_test_holder(self)   

    def is_leaf(self):
        return True

    def get_attributes(self):
        return self.attribute_header.get_attributes()

    def get_categories(self):
        return self.categories

    def get_matrix_train(self):
        return self.matrix_train_holder.get_matrix()

    def get_matrix_test(self):
        return self.matrix_test_holder.get_matrix()

    def get_train_files(self):
        return self.matrix_train_holder.get_instance_namefiles()
        #return self.corpus_file_list_train

    def get_test_files(self):
        return self.matrix_test_holder.get_instance_namefiles()
        #return self.corpus_file_list_test
        
    def get_train_instance_categories(self):
        return self.matrix_train_holder.get_instance_categories()
        #return self.corpus_file_list_train

    def get_test_instance_categories(self):
        return self.matrix_test_holder.get_instance_categories()
        #return self.corpus_file_list_test

    def get_tokens(self):
        pass

    def add(self, space_component):
        raise UnsupportedOperationError("Unsupported operation")

    def remove(self, space_component):
        raise UnsupportedOperationError("Unsupported operation")

    def get_child(self, i):
        raise UnsupportedOperationError("Unsupported operation")

    def create_iterator(self):
        return NullIterator()


class SpaceCompositeIterator(object):

    def __init__(self, iterator):
        self.stack = []
        self.stack.append(iterator)

    def __iter__(self):
        return self

    def next(self):

        if len(self.stack) == 0:
            raise StopIteration
        else:
            iterator = self.stack[-1]

            while(True):
                try:
                    e = iterator.next()
                except StopIteration:
                    self.stack.pop()
                    e = self.next()
                else:
                    if isinstance(e, SpaceComponent):
                        self.stack.append(e.create_iterator())
                        return e
                    else:
                        raise StopIteration
                return e


class NullIterator(object):

    def next(self):
        raise StopIteration

    def __iter__(self):
        return self


class UnsupportedOperationError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)
    
    
class NonIdenticalInstancesOfSubspacesError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

if __name__ == '__main__':

    print "You have to import this module!"