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
import subprocess
import multiprocessing

from gensim import corpora, models, similarities
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import mixture
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

from abc import ABCMeta, abstractmethod

import nltk
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.corpus.reader.tagged import CategorizedTaggedCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from ..attributes.attr_config import FactoryTermLex
from ..attributes import virtuals
from _pyio import __metaclass__
#from aptsources.distinfo import Template
from boto.ec2.cloudwatch.dimension import Dimension
from gensim.models.word2vec import Word2Vec
from gensim.matutils import Dense2Corpus
from numpy import transpose
from nltk.classify.util import log_likelihood
from ..attributes.virtuals \
import FilterTermsVirtualGlobalProcessor, FilterTermsVirtualReProcessor
from ..representations.extensions import FreqDistExt

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
    
    @staticmethod
    def perform_EM(target_mat, covariance_type, max_it):
    
        print "BEGINING EM ..."
        new_score=None
        best_score=None
        #best_score = numpy.infty #0.0
        good_clusterer = None
        for it in range(max_it)[1:]:
            print "ITERATION: " + str(it)
            
            clusterer = mixture.GMM(n_components=it, 
                                    covariance_type=covariance_type,
                                    n_iter=100,
                                    n_init=10,
                                    thresh=.000001,
                                    min_covar=.000001)
            # target_mat=numpy.transpose(submatrix_concepts_docs)
            clusterer.fit(target_mat)
            
            #print clusterer.predict_proba(target_mat)
            #new_score = numpy.sum(numpy.amax(clusterer.predict_proba(target_mat), axis=1))
            
            the_log_likelihoods, the_resposibilities = clusterer.score_samples(target_mat)
            new_score = the_log_likelihoods.mean()
            
            print "SC: ", new_score
            if best_score is not None:
                if numpy.abs(new_score - best_score) < .0001:
                    break
                else:
                    if new_score < best_score:
                        best_score = new_score
                        good_clusterer = clusterer
                    else:
                        break
            else:
                best_score=new_score
                good_clusterer = clusterer
                
        print best_score
        print good_clusterer
        return good_clusterer
    
    
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
    
    # FIXME: Here there are a deging software error: As you can see the class
    # CorpusCategorizedFromCatMap still receiving kwargs_corpus['cat_pattern']
    # which is not necessary!!!. We have to rename the paremeters in the cons-
    # tructor of the class Corpus. This is because the latter class, forces...
    # (because it specyfies the parameters in the constructor) to all subclasses
    # to use these parameters. The solution is could be: Just Leave more generic
    # parameters (anyway subclass will specify them, or send it the complete
    # kwargs object. 
    @staticmethod
    def build_filtered_postagged_corpus_from_cat_map(categories, kwargs_corpus):
        if 'corpus_path' in kwargs_corpus and\
        'corpus_pattern' in kwargs_corpus and\
        'file_pattern' in kwargs_corpus and\
        'cat_pattern' in kwargs_corpus:
            corpus = POSTaggedCorpusCategorizedFromCatMap(categories,
                                       kwargs_corpus['corpus_path'],
                                       kwargs_corpus['corpus_pattern'],
                                       kwargs_corpus['file_pattern'],
                                       kwargs_corpus['cat_pattern'],
                                       kwargs_corpus['cat_map'],)
        else:
            corpus = POSTaggedCorpusCategorizedFromCatMap(categories, kwargs_corpus['corpus_path'], kwargs_corpus['cat_map'])

        corpus = Util.decorate_corpus(corpus, kwargs_corpus['filters_corpus'])
        return corpus
    
    @staticmethod
    def build_filtered_postagged_corpus(categories, kwargs_corpus):
        if 'corpus_path' in kwargs_corpus and\
        'corpus_pattern' in kwargs_corpus and\
        'file_pattern' in kwargs_corpus and\
        'cat_pattern' in kwargs_corpus:
            corpus = POSTaggedCorpusCategorized(categories,
                                       kwargs_corpus['corpus_path'],
                                       kwargs_corpus['corpus_pattern'],
                                       kwargs_corpus['file_pattern'],
                                       kwargs_corpus['cat_pattern'])
        else:
            corpus = POSTaggedCorpusCategorized(categories, kwargs_corpus['corpus_path'])

        corpus = Util.decorate_corpus(corpus, kwargs_corpus['filters_corpus'])
        return corpus

    @staticmethod
    def get_tuples_from_fdist(fdist):
        the_tuples = []
        for key in fdist.keys_sorted():
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
    def decorate_matrix_holder(matrix_holder, space, kwargs_decorators_matrix_holder, dataset="???"):

        factory_simple_decorator_matrix_holder = FactorySimpleDecoratorMatrixHolder()
        for kwargs_decorator_matrix_holder in kwargs_decorators_matrix_holder:
            print kwargs_decorator_matrix_holder
            decorator_matrix_holder = \
            factory_simple_decorator_matrix_holder.build(kwargs_decorator_matrix_holder["type_decorator_matrix"],
                                                   kwargs_decorator_matrix_holder,
                                                   matrix_holder)  # all this arguments kwargs and matrix_holder necessaries???
            if dataset == "train":
                matrix_holder = decorator_matrix_holder.create_matrix_train_holder(matrix_holder, space)
            elif dataset == "test":
                matrix_holder = decorator_matrix_holder.create_matrix_test_holder(matrix_holder, space)
            else:
                print "ERROR: You need to specify the dataset" 
                raise UnsupportedOperationError("ERROR: You need to specify the dataset")
            
        return matrix_holder
    
    @staticmethod
    def decorate_attribute_header(attribute_header, space, kwargs_decorators_attribute_header, dataset="???"):
        
        factory_simple_decorator_attribute_header = FactorySimpleDecoratorAttributeHeader()
        for kwargs_decorator_attribute_header in kwargs_decorators_attribute_header:
            print "Params-begin: ----------------------"
            print kwargs_decorator_attribute_header
            print "Params-end: ----------------------"
            attribute_header = \
            factory_simple_decorator_attribute_header.build(kwargs_decorator_attribute_header["type_decorator_matrix"],
                                                   kwargs_decorator_attribute_header,
                                                   attribute_header)  # all this arguments kwargs and matrix_holder necessaries???
        
        print attribute_header.get_attributes()
        print "BLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        return attribute_header


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
    
    
    @staticmethod
    def configure_postagged_categorized_corpus_from_cat_map(corpus_path,
                                                  cat_map,
                                     user_corpus_path=r'corpora/(.*/.*)',
                                     files_pattern=r'.+/.+',
                                     categories_pattern=r'(.+)/.+'):

        match = re.match(user_corpus_path,
                         corpus_path)

        corpus = LazyCorpusLoader(match.group(1),
                                  CategorizedTaggedCorpusReader,
                                  files_pattern, cat_file=cat_map)
        return corpus

    @staticmethod
    def configure_postagged_categorized_corpus(corpus_path,
                                     user_corpus_path=r'corpora/(.*/.*)',
                                     files_pattern=r'.+/.+',
                                     categories_pattern=r'(.+)/.+'):

        match = re.match(user_corpus_path,
                         corpus_path)

        corpus = LazyCorpusLoader(match.group(1),
                                  CategorizedTaggedCorpusReader,
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
    
    
class POSTaggedCorpusCategorized(Corpus):

    def __init__(self,
                 categories,
                 corpus_path,
                 user_corpus_path=r'corpora/(.*/.*)',
                 files_pattern=r'.+/.+',
                 categories_pattern=r'(.+)/.+'):

        super(POSTaggedCorpusCategorized, self).__init__(categories,
                                                corpus_path,
                                                user_corpus_path,
                                                files_pattern,
                                                categories_pattern)


    def get_corpus(self):
        return self._corpus

    def build_corpus(self, corpus_path, user_corpus_path, files_pattern, categories_pattern):
        self._corpus = Util.configure_postagged_categorized_corpus(corpus_path, user_corpus_path, files_pattern, categories_pattern)
        self.__docs = self._corpus.fileids(categories = self._categories)

        # DEBUG: Uncomment if you want to see the selected documents
        # print "CORPUS_PATH: " + self._corpus_path
        # print "DOCUMENTS: " + str(self.__docs)
        # print "CATEGORIES: " + str(self._categories)

    def get_docs(self):
        return self.__docs
    

class POSTaggedCorpusCategorizedFromCatMap(Corpus):

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
        
        super(POSTaggedCorpusCategorizedFromCatMap, self).__init__(categories,
                                                corpus_path,
                                                user_corpus_path,
                                                files_pattern,
                                                categories_pattern)
        
        
        
        

    def get_corpus(self):
        return self._corpus

    def build_corpus(self, corpus_path, user_corpus_path, files_pattern, categories_pattern):
        self._corpus = Util.configure_postagged_categorized_corpus_from_cat_map(corpus_path, self.__cat_map, user_corpus_path, files_pattern)
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
        
        old_train_docs = set(old_train_docs)
        
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

            fd_vocabulary_cat = FreqDistExt(tokens_cat)

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
                FreqDistExt(dic_file_tokens[author_file])
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
     TRAIN_TEST_FROM_CAT_MAP,
     TRAIN_TEST_POSTAGGED,
     TRAIN_TEST_POSTAGGED_FROM_CAT_MAP) = range(5)


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
        elif option == EnumCommonTemplate.TRAIN_TEST_POSTAGGED:
            return TrainTestPOSTaggedCorpusTemplate()
        elif option == EnumCommonTemplate.TRAIN_TEST_POSTAGGED_FROM_CAT_MAP:
            return TrainTestPOSTaggedCorpusFromCatMapTemplate()


class CorpusTemplate(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def calc_corpus(self, corpus_object):
        pass

# ==============================================================================
# FIXME:
# FIXME:
# FIXME: WE NEED TO CREATE A FACTORY TO CREATE THE CORPUS OBJECTS. In this way
# each corpus template would have a factory creating the especific object that it needs.
# DO WE REALLY NEED THE TEMPLATE PATTERN TO BUILD ALL CORPUS???,... looks that each template
# HAS THE SAME THING!!!!. MAYBE A FACTORY JUST FOR THE CORPUS; WILL BE ENOUGHT.  
# AND MAYBE IT IS POSIBLE TO ELIMINATE THIS THINGS ABOUT TEMPLATE DESIGN PATTERN.

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
        

class TrainTestPOSTaggedCorpusTemplate(CorpusTemplate):

    def calc_corpus(self, corpus_object):
        corpus_object.filtered_train_corpus = \
        Util.build_filtered_postagged_corpus(corpus_object.categories,
                                            corpus_object.kwargs_corpus['train_corpus'])

        corpus_object.filtered_test_corpus = \
        Util.build_filtered_postagged_corpus(corpus_object.categories,
                                            corpus_object.kwargs_corpus['test_corpus'])
        

class TrainTestPOSTaggedCorpusFromCatMapTemplate(CorpusTemplate):

    def calc_corpus(self, corpus_object):
        corpus_object.filtered_train_corpus = \
        Util.build_filtered_postagged_corpus_from_cat_map(corpus_object.categories,
                                            corpus_object.kwargs_corpus['train_corpus'])

        corpus_object.filtered_test_corpus = \
        Util.build_filtered_postagged_corpus_from_cat_map(corpus_object.categories,
                                            corpus_object.kwargs_corpus['test_corpus'])

# ==============================================================================

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
     CSA,
     LSA,
     LDA,
     DOR,
     W2V,
     VW2V,
     SOA2,
     TFIDF,
     W2VVLAD,
     TCOR) = range(11)

class AttributeHeader(object):

    _metaclass_ = ABCMeta

    def __init__(self, fdist, vocabulary, concepts):
        self._fdist = fdist
        self._vocabulary = vocabulary
        self._concepts = concepts

    @abstractmethod
    def get_attributes(self):
        pass


class AttributeHeaderBOW(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderBOW, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        return self._vocabulary
    
class AttributeHeaderTFIDF(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderTFIDF, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        return self._vocabulary


class AttributeHeaderCSA(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderCSA, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        return self._concepts
    
    
class AttributeHeaderLSA(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderLSA, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        self.__str_concepts = []        
        for e in range(self._concepts):
            self.__str_concepts += ["c_" + str(e)]
            
        return self.__str_concepts
    
    
class AttributeHeaderDOR(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderDOR, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        self.__str_concepts = []        
        for e in range(len(self._concepts)):
            self.__str_concepts += ["document_" + str(e)]
            
        return self.__str_concepts
    
    
class AttributeHeaderTCOR(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderTCOR, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        self.__str_concepts = []        
        for e in range(len(self._concepts)):
            self.__str_concepts += ["cooccur_" + str(e)]
            
        return self.__str_concepts
    
    
class AttributeHeaderLDA(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderLDA, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        self.__str_concepts = []        
        for e in range(self._concepts):
            self.__str_concepts += ["c_" + str(e)]
            
        return self.__str_concepts
    
class AttributeHeaderW2V(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderW2V, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        self.__str_concepts = []        
        for e in range(self._concepts):
            self.__str_concepts += ["dimension_" + str(e)]
            
        return self.__str_concepts
    
class AttributeHeaderVW2V(AttributeHeader):

    def __init_(self, fdist, vocabulary, concepts):
        super(AttributeHeaderVW2V, self).__init__(fdist, vocabulary, concepts)

    def get_attributes(self):
        self.__str_concepts = []        
        for e in range(self._concepts):
            self.__str_concepts += ["dimension_" + str(e)]
            
        return self.__str_concepts


class FactoryDecoratorAttributeHeader(object):

    __metaclass__ = ABCMeta

    def build(self,option, kwargs, attribute_header): # all this arguments kwargs and matrix_holder necessaries???
        option = eval(option)
        return self.create(option, kwargs, attribute_header)

    @abstractmethod
    def create(self, option, kwargs, attribute_header): # all this arguments kwargs and matrix_holder necessaries???
        pass


class FactorySimpleDecoratorAttributeHeader(FactoryDecoratorAttributeHeader):

    def create(self, option, kwargs, attribute_header): # all this arguments kwargs and matrix_holder necessaries???
        if option == EnumDecoratorsMatrixHolder.FIXED_QUANTIZED:
            return FixedQuantizedAttributeHeader(attribute_header, kwargs["k_centers"]);
        if option == EnumDecoratorsMatrixHolder.FIXED_DISTANCES_TO_CLUSTER_TERMS:
            return FixedDistances2CTAttributeHeader(attribute_header, kwargs["k_centers"]);
        if option == EnumDecoratorsMatrixHolder.FIXED_DISTANCES_TO_CLUSTER_DOCS:
            return FixedDistances2CDAttributeHeader(attribute_header, kwargs["k_centers"]);
        if option == EnumDecoratorsMatrixHolder.NORM_SUM_ONE:
            # a TRANSPARENT attribute header
            return DecoratorAttributeHeader(attribute_header)
        if option == EnumDecoratorsMatrixHolder.DISTANCES_TO_CLUSTER_TERMS:
            return Distances2CTAttributeHeader(attribute_header, kwargs["k_centers"]);
        if option == EnumDecoratorsMatrixHolder.DISTANCES_TO_CLUSTER_DOCS:
            return Distances2CDAttributeHeader(attribute_header, kwargs["k_centers"]);
        if option == EnumDecoratorsMatrixHolder.FIXED_QUANTIZED_TFIDF:
            return FixedQuantizedTFIDFAttributeHeader(attribute_header, kwargs["k_centers"]);
            
    
class DecoratorAttributeHeader(AttributeHeader):
    
    def __init__(self, attribute_header):
        self.__attribute_header = attribute_header            

    def get_attributes(self):
        return self.__attribute_header.get_attributes()
        

class FixedQuantizedAttributeHeader(DecoratorAttributeHeader):
    
    def __init__(self, attribute_header, k):
        super(FixedQuantizedAttributeHeader, self).__init__(attribute_header)
        self.__k = k

    def get_attributes(self):
        old_attributes = super(FixedQuantizedAttributeHeader, self).get_attributes()
        
        self.__str_concepts = []        
        for e in range(self.__k):
            self.__str_concepts += ["prototype_" + str(e)]
            
        return self.__str_concepts
    
class FixedQuantizedTFIDFAttributeHeader(DecoratorAttributeHeader):
    
    def __init__(self, attribute_header, k):
        super(FixedQuantizedTFIDFAttributeHeader, self).__init__(attribute_header)
        self.__k = k

    def get_attributes(self):
        old_attributes = super(FixedQuantizedTFIDFAttributeHeader, self).get_attributes()
        
        self.__str_concepts = []        
        for e in range(self.__k):
            self.__str_concepts += ["prototype_" + str(e)]
            
        return self.__str_concepts
    

class FixedDistances2CTAttributeHeader(DecoratorAttributeHeader):
    
    def __init__(self, attribute_header, k):
        super(FixedDistances2CTAttributeHeader, self).__init__(attribute_header)
        self.__k = k

    def get_attributes(self):
        old_attributes = super(FixedDistances2CTAttributeHeader, self).get_attributes()
        
        self.__str_concepts = []        
        for e in range(self.__k):
            self.__str_concepts += ["prototype_" + str(e)]
            
        return self.__str_concepts
    
    
class FixedDistances2CDAttributeHeader(DecoratorAttributeHeader):
    
    def __init__(self, attribute_header, k):
        super(FixedDistances2CDAttributeHeader, self).__init__(attribute_header)
        self.__k = k

    def get_attributes(self):
        old_attributes = super(FixedDistances2CDAttributeHeader, self).get_attributes()
        
        self.__str_concepts = []        
        for e in range(self.__k):
            self.__str_concepts += ["prototype_" + str(e)]
            
        return self.__str_concepts
    
    
class Distances2CTAttributeHeader(DecoratorAttributeHeader):
    
    def __init__(self, attribute_header, k):
        super(Distances2CTAttributeHeader, self).__init__(attribute_header)
        self.__k = k

    def get_attributes(self):
        old_attributes = super(Distances2CTAttributeHeader, self).get_attributes()
        
        self.__str_concepts = []        
        for e in range(self.__k):
            self.__str_concepts += ["prototype_" + str(e)]
            
        return self.__str_concepts
    

class Distances2CDAttributeHeader(DecoratorAttributeHeader):
    
    def __init__(self, attribute_header, k):
        super(Distances2CDAttributeHeader, self).__init__(attribute_header)
        self.__k = k

    def get_attributes(self):
        old_attributes = super(Distances2CDAttributeHeader, self).get_attributes()
        
        self.__str_concepts = []        
        for e in range(self.__k):
            self.__str_concepts += ["prototype_" + str(e)]
            
        return self.__str_concepts
    
        
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
        
        if option == EnumRepresentation.LSA:
            return FactoryLSARepresentation()
        
        if option == EnumRepresentation.LDA:
            return FactoryLDARepresentation()
        
        if option == EnumRepresentation.DOR:
            return FactoryDORRepresentation()
        
        if option == EnumRepresentation.W2V:
            return FactoryW2VRepresentation()
        
        if option == EnumRepresentation.VW2V:
            return FactoryVW2VRepresentation()
        
        if option == EnumRepresentation.SOA2:
            return FactoryCSA2Representation()
        
        if option == EnumRepresentation.TFIDF:
            return FactoryTFIDFRepresentation()
        
        if option == EnumRepresentation.W2VVLAD:
            return FactoryW2VVLADRepresentation()
        
        if option == EnumRepresentation.TCOR:
            return FactoryTCORRepresentation()


class AbstractFactoryRepresentation(object):

    __metaclass__ = ABCMeta

    def build_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return self.create_attribute_header(fdist, vocabulary, concepts, space) 

    def build_matrix_train_holder(self, space):
        return self.create_matrix_train_holder(space)

    def build_matrix_test_holder(self, space):
        return self.create_matrix_test_holder(space)

    @abstractmethod
    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
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

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        self.__bow_attribute_header = AttributeHeaderBOW(fdist, vocabulary, concepts)
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__bow_attribute_header = Util.decorate_attribute_header(self.__bow_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__bow_attribute_header

    def create_matrix_train_holder(self, space):        
        self.__bow_train_matrix_holder = BOWTrainMatrixHolder(space) 
        
        # Decorating --------------------------------------------------
        # print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__bow_train_matrix_holder = Util.decorate_matrix_holder(self.__bow_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__bow_train_matrix_holder.build_matrix()
        return self.__bow_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        self.__bow_test_matrix_holder = BOWTestMatrixHolder(space) 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__bow_test_matrix_holder = Util.decorate_matrix_holder(self.__bow_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        self.__bow_test_matrix_holder.build_matrix()
        return self.__bow_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__bow_train_matrix_holder is not None:
            self.__bow_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__bow_train_matrix_holder = BOWTrainMatrixHolder(space)
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__bow_train_matrix_holder = Util.decorate_matrix_holder(self.__bow_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__bow_train_matrix_holder =  self.__bow_train_matrix_holder.load_train_data(space)
        return self.__bow_train_matrix_holder
    

class FactoryBOWRepresentation_bak(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return AttributeHeaderBOW(fdist, vocabulary, concepts)

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
            print "ERROR BOW: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        self.__bow_train_matrix_holder = BOWTrainMatrixHolder(space)        
        self.__bow_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_terms.npy"))
        self.__bow_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        self.__bow_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return self.__bow_train_matrix_holder     


class FactoryCSARepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):    
        self.__csa_attribute_header = AttributeHeaderCSA(fdist, vocabulary, concepts)
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__csa_attribute_header = Util.decorate_attribute_header(self.__csa_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__csa_attribute_header

    def create_matrix_train_holder(self, space):
        self.__csa_train_matrix_holder = CSATrainMatrixHolder(space)
        
        # Decorating --------------------------------------------------
        # print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__csa_train_matrix_holder = Util.decorate_matrix_holder(self.__csa_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__csa_train_matrix_holder.build_matrix()
        return self.__csa_train_matrix_holder

    def create_matrix_test_holder(self, space):
        self.__csa_test_matrix_holder = CSATestMatrixHolder(space, 
                                                            train_matrix_holder=self.__csa_train_matrix_holder)
                                                            #numpy.transpose(self.__csa_train_matrix_holder.get_matrix_terms()))
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__csa_test_matrix_holder = Util.decorate_matrix_holder(self.__csa_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        # Post initialization of what exactly the TEST object will need!.
        # The exact shared resource that the test_matrix_holder needs is CONTAINED in the 
        # last DECORATED version of the train_matrix_holder.
        # self.__csa_test_matrix_holder.set_shared_resource(self.__csa_train_matrix_holder.get_shared_resource())
        # -------------------------------------------------------------
        
        self.__csa_test_matrix_holder.build_matrix()
        return self.__csa_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__csa_train_matrix_holder is not None:
            self.__csa_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__csa_train_matrix_holder = CSATrainMatrixHolder(space)
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__csa_train_matrix_holder = Util.decorate_matrix_holder(self.__csa_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__csa_train_matrix_holder =  self.__csa_train_matrix_holder.load_train_data(space)
        return self.__csa_train_matrix_holder
    
    
class FactoryCSARepresentation_bak(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return AttributeHeaderCSA(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, space):
        self.__csa_train_matrix_holder = CSATrainMatrixHolder(space)
        self.__csa_train_matrix_holder.build_matrix()
        return self.__csa_train_matrix_holder

    def create_matrix_test_holder(self, space):
        self.__csa_test_matrix_holder = CSATestMatrixHolder(space, self.__csa_train_matrix_holder.get_matrix_terms_concepts())
        self.__csa_test_matrix_holder.build_matrix()
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
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        self.__csa_train_matrix_holder = CSATrainMatrixHolder(space)        
        self.__csa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))        
        self.__csa_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        self.__csa_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        self.__csa_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return self.__csa_train_matrix_holder     
    
    
class FactoryCSA2Representation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):    
        self.__csa2_attribute_header = AttributeHeaderCSA(fdist, vocabulary, concepts)
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################FALTA HEADER
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__csa2_attribute_header = Util.decorate_attribute_header(self.__csa2_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__csa2_attribute_header

    def create_matrix_train_holder(self, space):
        self.__csa2_train_matrix_holder = CSA2TrainMatrixHolder(space)
        
        # Decorating --------------------------------------------------
        # print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__csa2_train_matrix_holder = Util.decorate_matrix_holder(self.__csa2_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__csa2_train_matrix_holder.build_matrix()
        
        # END OF REBUILD THE HEADER USING SUBGROUPS LABELS
        space.attribute_header = \
        space.representation.build_attribute_header(space._fdist,
                                                    space._vocabulary,
                                                    self.__csa2_train_matrix_holder.get_ordered_new_labels_set(),
                                                    space)
        # END OF REBUILD THE HEADER USING SUBGROUPS LABELS
        
        return self.__csa2_train_matrix_holder

    def create_matrix_test_holder(self, space):
        self.__csa2_test_matrix_holder = CSA2TestMatrixHolder(space, 
                                                            train_matrix_holder=self.__csa2_train_matrix_holder)
                                                            #numpy.transpose(self.__csa_train_matrix_holder.get_matrix_terms()))
        
        self.__csa2_test_matrix_holder.set_dimensions_soa2(self.__csa2_train_matrix_holder.get_dimensions_soa2())
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__csa2_test_matrix_holder = Util.decorate_matrix_holder(self.__csa2_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        # Post initialization of what exactly the TEST object will need!.
        # The exact shared resource that the test_matrix_holder needs is CONTAINED in the 
        # last DECORATED version of the train_matrix_holder.
        # self.__csa_test_matrix_holder.set_shared_resource(self.__csa_train_matrix_holder.get_shared_resource())
        # -------------------------------------------------------------
        
        self.__csa2_test_matrix_holder.build_matrix()
        
        # END OF REBUILD THE HEADER USING SUBGROUPS LABELS
        space.attribute_header = \
        space.representation.build_attribute_header(space._fdist,
                                                    space._vocabulary,
                                                    self.__csa2_train_matrix_holder.get_ordered_new_labels_set(),
                                                    space)
        # END OF REBUILD THE HEADER USING SUBGROUPS LABELS
        
        return self.__csa2_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__csa2_train_matrix_holder is not None:
            self.__csa2_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__csa2_train_matrix_holder = CSA2TrainMatrixHolder(space)
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__csa2_train_matrix_holder = Util.decorate_matrix_holder(self.__csa2_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__csa2_train_matrix_holder =  self.__csa2_train_matrix_holder.load_train_data(space)
        return self.__csa2_train_matrix_holder
    
    
class FactoryLSARepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        self.__lsa_attribute_header = AttributeHeaderLSA(fdist, vocabulary, space.kwargs_space['concepts'])
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__lsa_attribute_header = Util.decorate_attribute_header(self.__lsa_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__lsa_attribute_header

    def create_matrix_train_holder(self, space):        
        self.__lsa_train_matrix_holder = LSATrainMatrixHolder(space, dataset_label="train") 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:  
            self.__lsa_train_matrix_holder = Util.decorate_matrix_holder(self.__lsa_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__lsa_train_matrix_holder.build_matrix()
        return self.__lsa_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        # self.__lsa_test_matrix_holder = LSATestMatrixHolder(space, self.__lsa_train_matrix_holder.get_id2word(), self.__lsa_train_matrix_holder.get_tfidf(), self.__lsa_train_matrix_holder.get_lsa(), "test")
        
        self.__lsa_test_matrix_holder = LSATestMatrixHolder(space,
                                                            dataset_label="test")

                                                            #train_matrix_holder=self.__lsa_train_matrix_holder, 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__lsa_test_matrix_holder = Util.decorate_matrix_holder(self.__lsa_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        self.__lsa_test_matrix_holder.build_matrix() 
        return self.__lsa_test_matrix_holder    
    
    def save_train_data(self, space):
        if self.__lsa_train_matrix_holder is not None:
            self.__lsa_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__lsa_train_matrix_holder = LSATrainMatrixHolder(space, dataset_label="train")
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__lsa_train_matrix_holder = Util.decorate_matrix_holder(self.__lsa_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__lsa_train_matrix_holder =  self.__lsa_train_matrix_holder.load_train_data(space)        
        return self.__lsa_train_matrix_holder 
    
    
class FactoryTFIDFRepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        self.__tfidf_attribute_header = AttributeHeaderTFIDF(fdist, vocabulary, space.kwargs_space['concepts'])
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__tfidf_attribute_header = Util.decorate_attribute_header(self.__tfidf_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__tfidf_attribute_header

    def create_matrix_train_holder(self, space):        
        self.__tfidf_train_matrix_holder = TFIDFTrainMatrixHolder(space, dataset_label="train") 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:  
            self.__tfidf_train_matrix_holder = Util.decorate_matrix_holder(self.__tfidf_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__tfidf_train_matrix_holder.build_matrix()
        return self.__tfidf_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        # self.__lsa_test_matrix_holder = LSATestMatrixHolder(space, self.__lsa_train_matrix_holder.get_id2word(), self.__lsa_train_matrix_holder.get_tfidf(), self.__lsa_train_matrix_holder.get_lsa(), "test")
        
        self.__tfidf_test_matrix_holder = TFIDFTestMatrixHolder(space,
                                                            dataset_label="test")

                                                            #train_matrix_holder=self.__lsa_train_matrix_holder, 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__tfidf_test_matrix_holder = Util.decorate_matrix_holder(self.__tfids_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        self.__tfidf_test_matrix_holder.build_matrix() 
        return self.__tfidf_test_matrix_holder    
    
    def save_train_data(self, space):
        if self.__tfidf_train_matrix_holder is not None:
            self.__tfidf_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__tfidf_train_matrix_holder = TFIDFTrainMatrixHolder(space, dataset_label="train")
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__tfidf_train_matrix_holder = Util.decorate_matrix_holder(self.__tfidf_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__tfidf_train_matrix_holder =  self.__tfidf_train_matrix_holder.load_train_data(space)        
        return self.__tfidf_train_matrix_holder 
    

class FactoryLSARepresentation_bak(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return AttributeHeaderLSA(fdist, vocabulary, space.kwargs_space['concepts'])

    def create_matrix_train_holder(self, space):        
        self.__lsa_train_matrix_holder = LSATrainMatrixHolder(space, dataset_label="train") 
        self.__lsa_train_matrix_holder.build_matrix()
        return self.__lsa_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        self.__lsa_test_matrix_holder = LSATestMatrixHolder(space, self.__lsa_train_matrix_holder.get_id2word(), self.__lsa_train_matrix_holder.get_tfidf(), self.__lsa_train_matrix_holder.get_lsa(), "test")
        self.__lsa_test_matrix_holder.build_matrix() 
        return self.__lsa_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__lsa_train_matrix_holder is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
            
            id2word = self.__lsa_train_matrix_holder.get_id2word()            
            with open(space.space_path + "/lsa/" + space.id_space + "_id2word.txt", 'w') as outfile:
                json.dump(id2word, outfile)  
                          
            tfidf = self.__lsa_train_matrix_holder.get_tfidf()
            tfidf.save(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")
            
            lsa = self.__lsa_train_matrix_holder.get_lsa()
            lsa.save(space.space_path + "/lsa/" + space.id_space + "_model.lsi") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.__lsa_train_matrix_holder.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.__lsa_train_matrix_holder.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.__lsa_train_matrix_holder.get_instance_categories())
        else:
            print "ERROR LSA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        self.__lsa_train_matrix_holder = LSATrainMatrixHolder(space)
        
        with open(space.space_path + "/lsa/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")      
        lsa = models.LsiModel.load(space.space_path + "/lsa/" + space.id_space + "_model.lsi")    
        
        self.__lsa_train_matrix_holder.set_id2word(id2word)
        self.__lsa_train_matrix_holder.set_tfidf(tfidf)
        self.__lsa_train_matrix_holder.set_lsa(lsa)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        self.__lsa_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        self.__lsa_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        self.__lsa_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return self.__lsa_train_matrix_holder   
    
    
class FactoryW2VRepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        self.__w2v_attribute_header = AttributeHeaderW2V(fdist, vocabulary, space.kwargs_space['concepts'])
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__w2v_attribute_header = Util.decorate_attribute_header(self.__w2v_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__w2v_attribute_header

    def create_matrix_train_holder(self, space):        
        self.__w2v_train_matrix_holder = W2VTrainMatrixHolder(space, dataset_label="train") 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:  
            self.__w2v_train_matrix_holder = Util.decorate_matrix_holder(self.__w2v_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__w2v_train_matrix_holder.build_matrix()
        return self.__w2v_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        # self.__lsa_test_matrix_holder = LSATestMatrixHolder(space, self.__lsa_train_matrix_holder.get_id2word(), self.__lsa_train_matrix_holder.get_tfidf(), self.__lsa_train_matrix_holder.get_lsa(), "test")
        
        self.__w2v_test_matrix_holder = W2VTestMatrixHolder(space,
                                                            dataset_label="test")

                                                            #train_matrix_holder=self.__lsa_train_matrix_holder, 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__w2v_test_matrix_holder = Util.decorate_matrix_holder(self.__w2v_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        self.__w2v_test_matrix_holder.build_matrix() 
        return self.__w2v_test_matrix_holder    
    
    def save_train_data(self, space):
        if self.__w2v_train_matrix_holder is not None:
            self.__w2v_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR W2V: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__w2v_train_matrix_holder = W2VTrainMatrixHolder(space, dataset_label="train")
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__w2v_train_matrix_holder = Util.decorate_matrix_holder(self.__w2v_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__w2v_train_matrix_holder =  self.__w2v_train_matrix_holder.load_train_data(space)        
        return self.__w2v_train_matrix_holder 
    

class FactoryVW2VRepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        self.__vw2v_attribute_header = AttributeHeaderVW2V(fdist, vocabulary, space.kwargs_space['concepts'])
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__vw2v_attribute_header = Util.decorate_attribute_header(self.__vw2v_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__vw2v_attribute_header

    def create_matrix_train_holder(self, space):        
        self.__vw2v_train_matrix_holder = VW2VTrainMatrixHolder(space, dataset_label="train") 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:  
            self.__vw2v_train_matrix_holder = Util.decorate_matrix_holder(self.__vw2v_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__vw2v_train_matrix_holder.build_matrix()
        return self.__vw2v_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        # self.__lsa_test_matrix_holder = LSATestMatrixHolder(space, self.__lsa_train_matrix_holder.get_id2word(), self.__lsa_train_matrix_holder.get_tfidf(), self.__lsa_train_matrix_holder.get_lsa(), "test")
        
        self.__vw2v_test_matrix_holder = VW2VTestMatrixHolder(space,
                                                            dataset_label="test")

                                                            #train_matrix_holder=self.__lsa_train_matrix_holder, 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__vw2v_test_matrix_holder = Util.decorate_matrix_holder(self.__vw2v_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        self.__vw2v_test_matrix_holder.build_matrix() 
        return self.__vw2v_test_matrix_holder    
    
    def save_train_data(self, space):
        if self.__vw2v_train_matrix_holder is not None:
            self.__vw2v_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR W2V: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__vw2v_train_matrix_holder = VW2VTrainMatrixHolder(space, dataset_label="train", train=False)
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__vw2v_train_matrix_holder = Util.decorate_matrix_holder(self.__vw2v_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__vw2v_train_matrix_holder =  self.__vw2v_train_matrix_holder.load_train_data(space)        
        return self.__vw2v_train_matrix_holder 
        
    
class FactoryDORRepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):   
        
        self.__dor_attribute_header = AttributeHeaderDOR(fdist, vocabulary, space.corpus_file_list_train)
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__dor_attribute_header = Util.decorate_attribute_header(self.__dor_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__dor_attribute_header

    def create_matrix_train_holder(self, space):  
        
        # Basic initialization ----------------------------------------      
        self.__dor_train_matrix_holder = DORTrainMatrixHolder(space, 
                                                              dataset_label="train")
        self.__dor_train_matrix_holder.build_bowcorpus_id2word(space, 
                                                               space.virtual_classes_holder_train, 
                                                               space.corpus_file_list_train)
        # Basic initialization ----------------------------------------
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__dor_train_matrix_holder = Util.decorate_matrix_holder(self.__dor_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        # Post initialization of what exactly the TEST object will need!.
        # The exact shared resource that the test_matrix_holder needs is CONTAINED in the 
        # last DECORATED version of the train_matrix_holder.
        self.__dor_train_matrix_holder.build_matrix()
        # -------------------------------------------------------------        
        return self.__dor_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        #self.__dor_test_matrix_holder = DORTestMatrixHolder(space, self.__dor_train_matrix_holder.get_id2word(), self.__dor_train_matrix_holder.get_tfidf(), self.__dor_train_matrix_holder.get_lsa(), "test")
        #self.__dor_test_matrix_holder.set_mat_docs_terms(self.__dor_train_matrix_holder.get_mat_docs_terms())
        
        # These are the base initialization. Methods like set_matrix_terms, never should be overwriten by decorators
        
        # Basic initialization ----------------------------------------
        self.__dor_test_matrix_holder = DORTestMatrixHolder(space, 
                                                            train_matrix_holder=self.__dor_train_matrix_holder,
                                                            dataset_label="test")
        #self.__dor_test_matrix_holder.set_matrix_terms(self.__dor_train_matrix_holder.get_matrix_terms())
        # Basic initialization ----------------------------------------
                
        #self.__dor_test_matrix_holder.set_shared_resource(self.__dor_train_matrix_holder.get_shared_resource())
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__dor_test_matrix_holder = Util.decorate_matrix_holder(self.__dor_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        # Post initialization of what exactly the TEST object will need!.
        # The exact shared resource that the test_matrix_holder needs is CONTAINED in the 
        # last DECORATED version of the train_matrix_holder.
        # self.__dor_test_matrix_holder.set_shared_resource(self.__dor_train_matrix_holder.get_shared_resource())
        self.__dor_test_matrix_holder.build_matrix() 
        # -------------------------------------------------------------
        return self.__dor_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__dor_train_matrix_holder is not None:
            self.__dor_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR DOR: There is not a train matrix terms concepts built"
        
#         
#         if self.__dor_train_matrix_holder is not None:
#             cache_file = "%s/%s" % (space.space_path, space.id_space)
#             
#             #numpy.save(cache_file + "_mat_terms_concepts.npy", 
#             #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
#             
#             id2word = self.__dor_train_matrix_holder.get_id2word()            
#             with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'w') as outfile:
#                 json.dump(id2word, outfile)  
#                           
#             #tfidf = self.__dor_train_matrix_holder.get_tfidf()
#             #tfidf.save(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")
#             
#             #dor = self.__dor_train_matrix_holder.get_dor()
#             #dor.save(space.space_path + "/dor/" + space.id_space + "_model.dor") # same for tfidf, lda, ...
#             
#             
#             numpy.save(cache_file + "_mat_docs_docs.npy", 
#                        self.__dor_train_matrix_holder.get_matrix())
#             
#             numpy.save(cache_file + "_mat_docs_terms.npy", 
#                        self.__dor_train_matrix_holder.get_mat_docs_terms())
#             
#             numpy.save(cache_file + "_instance_namefiles.npy", 
#                        self.__dor_train_matrix_holder.get_instance_namefiles())
#             
#             numpy.save(cache_file + "_instance_categories.npy", 
#                        self.__dor_train_matrix_holder.get_instance_categories())
#         else:
#             print "ERROR LSA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        self.__dor_train_matrix_holder = DORTrainMatrixHolder(space, dataset_label="train")
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__dor_train_matrix_holder = Util.decorate_matrix_holder(self.__dor_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__dor_train_matrix_holder =  self.__dor_train_matrix_holder.load_train_data(space)
        return self.__dor_train_matrix_holder
        
#         cache_file = "%s/%s" % (space.space_path, space.id_space)
#         
#         self.__dor_train_matrix_holder = DORTrainMatrixHolder(space)
#         
#         with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'r') as infile:
#                 id2word = json.load(infile)       
#         #tfidf = models.TfidfModel.load(space.space_path + "/dor/" + space.id_space + "_model.tfidf")      
#         #dor = models.LsiModel.load(space.space_path + "/dor/" + space.id_space + "_model.dor")    
#         
#         self.__dor_train_matrix_holder.set_id2word(id2word)
#         #self.__dor_train_matrix_holder.set_tfidf(tfidf)
#         #self.__dor_train_matrix_holder.set_dor(dor)
#           
#         #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
#         self.__dor_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_docs.npy"))
#         self.__dor_train_matrix_holder.set_mat_docs_terms(numpy.load(cache_file + "_mat_docs_terms.npy"))
#         self.__dor_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
#         self.__dor_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
#         
#         return self.__dor_train_matrix_holder   


class FactoryTCORRepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):   
        
        self.__tcor_attribute_header = AttributeHeaderTCOR(fdist, vocabulary, vocabulary)
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__tcor_attribute_header = Util.decorate_attribute_header(self.__tcor_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__tcor_attribute_header

    def create_matrix_train_holder(self, space):  
        
        # Basic initialization ----------------------------------------      
        self.__tcor_train_matrix_holder = TCORTrainMatrixHolder(space, 
                                                              dataset_label="train")
        self.__tcor_train_matrix_holder.build_bowcorpus_id2word(space, 
                                                               space.virtual_classes_holder_train, 
                                                               space.corpus_file_list_train)
        # Basic initialization ----------------------------------------
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__tcor_train_matrix_holder = Util.decorate_matrix_holder(self.__tcor_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        # Post initialization of what exactly the TEST object will need!.
        # The exact shared resource that the test_matrix_holder needs is CONTAINED in the 
        # last DECORATED version of the train_matrix_holder.
        self.__tcor_train_matrix_holder.build_matrix()
        # -------------------------------------------------------------        
        return self.__tcor_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        #self.__dor_test_matrix_holder = DORTestMatrixHolder(space, self.__dor_train_matrix_holder.get_id2word(), self.__dor_train_matrix_holder.get_tfidf(), self.__dor_train_matrix_holder.get_lsa(), "test")
        #self.__dor_test_matrix_holder.set_mat_docs_terms(self.__dor_train_matrix_holder.get_mat_docs_terms())
        
        # These are the base initialization. Methods like set_matrix_terms, never should be overwriten by decorators
        
        # Basic initialization ----------------------------------------
        self.__tcor_test_matrix_holder = TCORTestMatrixHolder(space, 
                                                            train_matrix_holder=self.__tcor_train_matrix_holder,
                                                            dataset_label="test")
        #self.__dor_test_matrix_holder.set_matrix_terms(self.__dor_train_matrix_holder.get_matrix_terms())
        # Basic initialization ----------------------------------------
                
        #self.__dor_test_matrix_holder.set_shared_resource(self.__dor_train_matrix_holder.get_shared_resource())
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__tcor_test_matrix_holder = Util.decorate_matrix_holder(self.__tcor_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        # Post initialization of what exactly the TEST object will need!.
        # The exact shared resource that the test_matrix_holder needs is CONTAINED in the 
        # last DECORATED version of the train_matrix_holder.
        # self.__dor_test_matrix_holder.set_shared_resource(self.__dor_train_matrix_holder.get_shared_resource())
        self.__tcor_test_matrix_holder.build_matrix() 
        # -------------------------------------------------------------
        return self.__tcor_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__tcor_train_matrix_holder is not None:
            self.__tcor_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR DOR: There is not a train matrix terms concepts built"
        
#         
#         if self.__dor_train_matrix_holder is not None:
#             cache_file = "%s/%s" % (space.space_path, space.id_space)
#             
#             #numpy.save(cache_file + "_mat_terms_concepts.npy", 
#             #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
#             
#             id2word = self.__dor_train_matrix_holder.get_id2word()            
#             with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'w') as outfile:
#                 json.dump(id2word, outfile)  
#                           
#             #tfidf = self.__dor_train_matrix_holder.get_tfidf()
#             #tfidf.save(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")
#             
#             #dor = self.__dor_train_matrix_holder.get_dor()
#             #dor.save(space.space_path + "/dor/" + space.id_space + "_model.dor") # same for tfidf, lda, ...
#             
#             
#             numpy.save(cache_file + "_mat_docs_docs.npy", 
#                        self.__dor_train_matrix_holder.get_matrix())
#             
#             numpy.save(cache_file + "_mat_docs_terms.npy", 
#                        self.__dor_train_matrix_holder.get_mat_docs_terms())
#             
#             numpy.save(cache_file + "_instance_namefiles.npy", 
#                        self.__dor_train_matrix_holder.get_instance_namefiles())
#             
#             numpy.save(cache_file + "_instance_categories.npy", 
#                        self.__dor_train_matrix_holder.get_instance_categories())
#         else:
#             print "ERROR LSA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        self.__tcor_train_matrix_holder = TCORTrainMatrixHolder(space, dataset_label="train")
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__tcor_train_matrix_holder = Util.decorate_matrix_holder(self.__tcor_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__tcor_train_matrix_holder =  self.__tcor_train_matrix_holder.load_train_data(space)
        return self.__tcor_train_matrix_holder
        
#         cache_file = "%s/%s" % (space.space_path, space.id_space)
#         
#         self.__dor_train_matrix_holder = DORTrainMatrixHolder(space)
#         
#         with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'r') as infile:
#                 id2word = json.load(infile)       
#         #tfidf = models.TfidfModel.load(space.space_path + "/dor/" + space.id_space + "_model.tfidf")      
#         #dor = models.LsiModel.load(space.space_path + "/dor/" + space.id_space + "_model.dor")    
#         
#         self.__dor_train_matrix_holder.set_id2word(id2word)
#         #self.__dor_train_matrix_holder.set_tfidf(tfidf)
#         #self.__dor_train_matrix_holder.set_dor(dor)
#           
#         #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
#         self.__dor_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_docs.npy"))
#         self.__dor_train_matrix_holder.set_mat_docs_terms(numpy.load(cache_file + "_mat_docs_terms.npy"))
#         self.__dor_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
#         self.__dor_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
#         
#         return self.__dor_train_matrix_holder 


class FactoryLDARepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        lda_attribute_header = AttributeHeaderLDA(fdist, vocabulary, space.kwargs_space['concepts'])
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            lda_attribute_header = Util.decorate_attribute_header(lda_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return lda_attribute_header

    def create_matrix_train_holder(self, space):        
        self.__lda_train_matrix_holder = LDATrainMatrixHolder(space, dataset_label="train") 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:  
            self.__lda_train_matrix_holder = Util.decorate_matrix_holder(self.__lda_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__lda_train_matrix_holder.build_matrix()
        return self.__lda_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        self.__lda_test_matrix_holder = LDATestMatrixHolder(space, self.__lda_train_matrix_holder.get_id2word(), 
                                                            self.__lda_train_matrix_holder.get_tfidf(), 
                                                            self.__lda_train_matrix_holder.get_lda(), 
                                                            dataset_label="test")
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__lda_test_matrix_holder = Util.decorate_matrix_holder(self.__lda_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        self.__lda_test_matrix_holder.build_matrix() 
        return self.__lda_test_matrix_holder
    
    def save_train_data(self, space):
        if self.__lda_train_matrix_holder is not None:
            self.__lda_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR LDA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__lda_train_matrix_holder = LDATrainMatrixHolder(space, dataset_label="train")
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__lda_train_matrix_holder = Util.decorate_matrix_holder(self.__lda_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__lda_train_matrix_holder =  self.__lda_train_matrix_holder.load_train_data(space)        
        return self.__lda_train_matrix_holder 


class FactoryLDARepresentation_bak(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return AttributeHeaderLDA(fdist, vocabulary, space.kwargs_space['concepts'])

    def create_matrix_train_holder(self, space):        
        self.__lda_train_matrix_holder = LDATrainMatrixHolder(space, dataset_label="train") 
        self.__lda_train_matrix_holder.build_matrix()
        return self.__lda_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        self.__lda_test_matrix_holder = LDATestMatrixHolder(space, self.__lda_train_matrix_holder.get_id2word(), self.__lda_train_matrix_holder.get_tfidf(), self.__lda_train_matrix_holder.get_lda(), "test")
        self.__lda_test_matrix_holder.build_matrix() 
        return self.__lda_test_matrix_holder
    
    def save_train_data(self, space):
        
        if self.__lda_train_matrix_holder is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lda_train_matrix_holder.get_matrix_terms_concepts())
            
            id2word = self.__lda_train_matrix_holder.get_id2word()            
            with open(space.space_path + "/lda/" + space.id_space + "_id2word.txt", 'w') as outfile:
                json.dump(id2word, outfile)  
                          
            tfidf = self.__lda_train_matrix_holder.get_tfidf()
            tfidf.save(space.space_path + "/lda/" + space.id_space + "_model.tfidf")
            
            lda = self.__lda_train_matrix_holder.get_lda()
            lda.save(space.space_path + "/lda/" + space.id_space + "_model.lda") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.__lda_train_matrix_holder.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.__lda_train_matrix_holder.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.__lda_train_matrix_holder.get_instance_categories())
        else:
            print "ERROR LDA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        self.__lda_train_matrix_holder = LDATrainMatrixHolder(space)
        
        with open(space.space_path + "/lda/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/lda/" + space.id_space + "_model.tfidf")      
        lda = models.LdaModel.load(space.space_path + "/lda/" + space.id_space + "_model.lda")    
        
        self.__lda_train_matrix_holder.set_id2word(id2word)
        self.__lda_train_matrix_holder.set_tfidf(tfidf)
        self.__lda_train_matrix_holder.set_lda(lda)
          
        #self.__lda_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        self.__lda_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        self.__lda_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        self.__lda_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return self.__lda_train_matrix_holder 
    

    
class FactoryW2VVLADRepresentation(AbstractFactoryRepresentation):

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        self.__w2vvlad_attribute_header = AttributeHeaderW2V(fdist, vocabulary, space.kwargs_space['concepts'] * space.kwargs_space['vlad_dim'])

        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__w2vvlad_attribute_header = Util.decorate_attribute_header(self.__w2vvlad_attribute_header,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'])
        # Decorating --------------------------------------------------      
           
        return self.__w2vvlad_attribute_header

    def create_matrix_train_holder(self, space):        
        self.__w2vvlad_train_matrix_holder = W2VVLADTrainMatrixHolder(space, dataset_label="train") 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:  
            self.__w2vvlad_train_matrix_holder = Util.decorate_matrix_holder(self.__w2vvlad_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__w2vvlad_train_matrix_holder.build_matrix()
        return self.__w2vvlad_train_matrix_holder 

    def create_matrix_test_holder(self,space):
        # self.__lsa_test_matrix_holder = LSATestMatrixHolder(space, self.__lsa_train_matrix_holder.get_id2word(), self.__lsa_train_matrix_holder.get_tfidf(), self.__lsa_train_matrix_holder.get_lsa(), "test")
        
        self.__w2vvlad_test_matrix_holder = W2VVLADTestMatrixHolder(space,
                                                            dataset_label="test")

                                                            #train_matrix_holder=self.__lsa_train_matrix_holder, 
        
        # Decorating --------------------------------------------------
        if 'decorators_matrix' in space.kwargs_space:   
            self.__w2vvlad_test_matrix_holder = Util.decorate_matrix_holder(self.__w2vvlad_test_matrix_holder,
                                                                        space,  
                                                                        space.kwargs_space['decorators_matrix'],
                                                                        "test")
        # Decorating --------------------------------------------------
        
        self.__w2vvlad_test_matrix_holder.build_matrix() 
        return self.__w2vvlad_test_matrix_holder    
    
    def save_train_data(self, space):
        if self.__w2vvlad_train_matrix_holder is not None:
            self.__w2vvlad_train_matrix_holder.save_train_data(space)
        else:
            print "ERROR W2V: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        self.__w2vvlad_train_matrix_holder = W2VVLADTrainMatrixHolder(space, dataset_label="train")
        
        # Decorating --------------------------------------------------
        print space.kwargs_space
        if 'decorators_matrix' in space.kwargs_space:  
            self.__w2vvlad_train_matrix_holder = Util.decorate_matrix_holder(self.__w2vvlad_train_matrix_holder,
                                                                     space, 
                                                                     space.kwargs_space['decorators_matrix'],
                                                                     "train")
        # Decorating --------------------------------------------------
        
        self.__w2vvlad_train_matrix_holder =  self.__w2vvlad_train_matrix_holder.load_train_data(space)        
        return self.__w2vvlad_train_matrix_holder 


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
    
    @abstractmethod
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    @abstractmethod
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    @abstractmethod
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    @abstractmethod
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    @abstractmethod
    def save_train_data(self, space):
        pass
    
    @abstractmethod
    def load_train_data(self, space):
        pass
    

class EnumDecoratorsMatrixHolder(object):

    (TRANSPARENT,
     FIXED_QUANTIZED,
     CLASS_DESCOMPOSITION,
     NORM_SUM_ONE,
     NORM_Z_SCORE,
     NORM_TFIDF,
     FIXED_DISTANCES_TO_CLUSTER_TERMS,
     FIXED_DISTANCES_TO_CLUSTER_DOCS,
     DISTANCES_TO_CLUSTER_TERMS,
     DISTANCES_TO_CLUSTER_DOCS,
     FIXED_QUANTIZED_TFIDF) = range(11)


class FactoryDecoratorMatrixHolder(object):

    __metaclass__ = ABCMeta

    def build(self,option, kwargs, matrix_holder): # all this arguments kwargs and matrix_holder necessaries???
        option = eval(option)
        return self.create(option, kwargs, matrix_holder)

    @abstractmethod
    def create(self, option, kwargs, matrix_holder): # all this arguments kwargs and matrix_holder necessaries???
        pass


class FactorySimpleDecoratorMatrixHolder(FactoryDecoratorMatrixHolder):

    def create(self, option, kwargs, matrix_holder): # all this arguments kwargs and matrix_holder necessaries???
        if option == EnumDecoratorsMatrixHolder.FIXED_QUANTIZED:
            if "precomputed_dict" in kwargs:
                pc = kwargs["precomputed_dict"]
            else:
                pc = "NO_PRECOMPUTED"
            return FactoryQuantizedDecoratorMatrixHolder(k_centers=kwargs["k_centers"], precomputed_dict=pc);
        
        if option == EnumDecoratorsMatrixHolder.NORM_SUM_ONE:
            return FactoryNormalizedProbsDecoratorMatrixHolder();
        
        if option == EnumDecoratorsMatrixHolder.FIXED_DISTANCES_TO_CLUSTER_TERMS:
            if "precomputed_dict" in kwargs:
                pc = kwargs["precomputed_dict"]
            else:
                pc = "NO_PRECOMPUTED"
                
            if "distance" in kwargs:
                dist = kwargs["distance"]
            else:
                dist = "euclidean"
            return FactoryFixedDistances2CTDecoratorMatrixHolder(k_centers=kwargs["k_centers"], precomputed_dict=pc, distance=dist);
        
        if option == EnumDecoratorsMatrixHolder.FIXED_DISTANCES_TO_CLUSTER_DOCS:
            if "precomputed_dict" in kwargs:
                pc = kwargs["precomputed_dict"]
            else:
                pc = "NO_PRECOMPUTED"
            return FactoryFixedDistances2CDDecoratorMatrixHolder(k_centers=kwargs["k_centers"], precomputed_dict=pc);

        if option == EnumDecoratorsMatrixHolder.DISTANCES_TO_CLUSTER_TERMS:
            if "precomputed_dict" in kwargs:
                pc = kwargs["precomputed_dict"]
            else:
                pc = "NO_PRECOMPUTED"
            return FactoryDistances2CTDecoratorMatrixHolder(k_centers=kwargs["k_centers"], precomputed_dict=pc);
        
        if option == EnumDecoratorsMatrixHolder.DISTANCES_TO_CLUSTER_DOCS:
            if "precomputed_dict" in kwargs:
                pc = kwargs["precomputed_dict"]
            else:
                pc = "NO_PRECOMPUTED"
            return FactoryDistances2CDDecoratorMatrixHolder(k_centers=kwargs["k_centers"], precomputed_dict=pc);
        
        if option == EnumDecoratorsMatrixHolder.FIXED_QUANTIZED_TFIDF:
            if "precomputed_dict" in kwargs:
                pc = kwargs["precomputed_dict"]
            else:
                pc = "NO_PRECOMPUTED"
            return FactoryQuantizedTFIDFDecoratorMatrixHolder(k_centers=kwargs["k_centers"], precomputed_dict=pc);
        
        #FIXED_QUANTIZED_TFIDF

class AbstractFactoryDecoratorMatrixHolder(object):

    __metaclass__ = ABCMeta

    def build_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return self.create_attribute_header(fdist, vocabulary, concepts, space) 

    def build_matrix_train_holder(self, matrix_holder, space):
        return self.create_matrix_train_holder(space)

    def build_matrix_test_holder(self, matrix_holder, space):
        return self.create_matrix_test_holder(space)

    @abstractmethod
    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
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

class FactoryQuantizedDecoratorMatrixHolder(AbstractFactoryDecoratorMatrixHolder):
    
    def __init__(self, k_centers=300, precomputed_dict="NO_PRECOMPUTED"):
        self.k_centers = k_centers
        self.precomputed_dict = precomputed_dict

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return FixedQuantizedAttributeHeader(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, matrix_holder, space):        
        return FixedQuantizedTrainMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)

    def create_matrix_test_holder(self, matrix_holder, space):
        return FixedQuantizedTestMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    

class FactoryQuantizedTFIDFDecoratorMatrixHolder(AbstractFactoryDecoratorMatrixHolder):
    
    def __init__(self, k_centers=300, precomputed_dict="NO_PRECOMPUTED"):
        self.k_centers = k_centers
        self.precomputed_dict = precomputed_dict

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return FixedQuantizedTFIDFAttributeHeader(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, matrix_holder, space):        
        return FixedQuantizedTFIDFTrainMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)

    def create_matrix_test_holder(self, matrix_holder, space):
        return FixedQuantizedTFIDFTestMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    
    
class FactoryNormalizedProbsDecoratorMatrixHolder(AbstractFactoryDecoratorMatrixHolder):
    
    def __init__(self):
        pass

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return DecoratorAttributeHeader(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, matrix_holder, space):        
        return NormalizedProbsDecoratorTrainMatrixHolder(matrix_holder)

    def create_matrix_test_holder(self, matrix_holder, space):
        return NormalizedProbsDecoratorTestMatrixHolder(matrix_holder)
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    
    
class FactoryFixedDistances2CTDecoratorMatrixHolder(AbstractFactoryDecoratorMatrixHolder):
    
    def __init__(self, k_centers=300, precomputed_dict="NO_PRECOMPUTED", distance="euclidean"):
        self.k_centers = k_centers
        self.precomputed_dict = precomputed_dict
        self.distance = distance

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return FixedDistances2CTAttributeHeader(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, matrix_holder, space):        
        return FixedDistances2CTTrainMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict, self.distance)

    def create_matrix_test_holder(self, matrix_holder, space):
        return FixedDistances2CTTestMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict, self.distance)
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass


class FactoryFixedDistances2CDDecoratorMatrixHolder(AbstractFactoryDecoratorMatrixHolder):
    
    def __init__(self, k_centers=300, precomputed_dict="NO_PRECOMPUTED"):
        self.k_centers = k_centers
        self.precomputed_dict = precomputed_dict

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return FixedDistances2CDAttributeHeader(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, matrix_holder, space):        
        return FixedDistances2CDTrainMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)

    def create_matrix_test_holder(self, matrix_holder, space):
        return FixedDistances2CDTestMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    

class FactoryDistances2CTDecoratorMatrixHolder(AbstractFactoryDecoratorMatrixHolder):
    
    def __init__(self, k_centers=300, precomputed_dict="NO_PRECOMPUTED"):
        self.k_centers = k_centers
        self.precomputed_dict = precomputed_dict

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return Distances2CTAttributeHeader(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, matrix_holder, space):        
        return Distances2CTTrainMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)

    def create_matrix_test_holder(self, matrix_holder, space):
        return Distances2CTTestMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    
    
class FactoryDistances2CDDecoratorMatrixHolder(AbstractFactoryDecoratorMatrixHolder):
    
    def __init__(self, k_centers=300, precomputed_dict="NO_PRECOMPUTED"):
        self.k_centers = k_centers
        self.precomputed_dict = precomputed_dict

    def create_attribute_header(self, fdist, vocabulary, concepts, space=None):
        return Distances2CDAttributeHeader(fdist, vocabulary, concepts)

    def create_matrix_train_holder(self, matrix_holder, space):        
        return Distances2CDTrainMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)

    def create_matrix_test_holder(self, matrix_holder, space):
        return Distances2CDTestMatrixHolder(matrix_holder, self.k_centers, self.precomputed_dict)
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    
    
class DecoratorMatrixHolder(MatrixHolder):
    
    def __init__(self, matrix_holder_object):
        self.__matrix_holder_object = matrix_holder_object
        
    def get_space(self):
        return self.__matrix_holder_object.space
        
    def set_space(self, value):   
        self.__matrix_holder_object = value         

    def normalize_matrix(self, normalizer, matrix):
        self.__matrix_holder_object.normalize_matrix()
    
    def get_instance_categories(self):
        return self.__matrix_holder_object.get_instance_categories()
    
    def get_instance_namefiles(self):
        return self.__matrix_holder_object.get_instance_namefiles()
    
    def set_instance_categories(self, value):
        self.__matrix_holder_object.set_instance_categories(value)
    
    def set_instance_namefiles(self, value):
        self.__matrix_holder_object.set_instance_namefiles(value)
        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return self.__matrix_holder_object.get_matrix_terms()
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self.__matrix_holder_object.set_matrix_terms(value)
        
    # THESE ARE THE BASIC THINGS THAT THE DECORATOR ADDS FUNCTIONALITY
        
    def build_matrix(self):
        self.__matrix_holder_object.build_matrix()    
        
    def get_matrix(self):
        return self.__matrix_holder_object.get_matrix()

    def set_matrix(self, value):
        self.__matrix_holder_object.set_matrix(value)       
     
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        return self.__matrix_holder_object.get_shared_resource()
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        self.__matrix_holder_object.set_shared_resource(value)
      
    def save_train_data(self, space):
        self.__matrix_holder_object.save_train_data(space)
    
    def load_train_data(self, space):
        return self.__matrix_holder_object.load_train_data(space)


class FixedQuantizedMatrixHolder(DecoratorMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedQuantizedMatrixHolder, self).__init__(matrix_holder)
        self.__k = k
        #self.term_matrix = term_matrix
        self.__clusterer = None
        self._precomputed_dict=precomputed_dict
        self._matrix = None
        
    def get_k_centers(self):
        return self.__k
    
    def set_k_centers(self, value):
        self.__k = value
        
    def compute_prototypes(self, matrix_terms):
        k= self.__k
        
        print "Begining clustering."
        if self._precomputed_dict == "NO_PRECOMPUTED":
            clusterer = KMeans(n_clusters=k, verbose=1, n_jobs=4)
            clusterer.fit(matrix_terms)
        else:
            cache_file = self._precomputed_dict        
            clusterer =  joblib.load(cache_file)
            
        print "End of clustering."
        
        self.set_shared_resource(clusterer)


    def build_matrix_quantized(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_terms):

        t1 = time.time()
        print "Starting BOW representation..."
        
        k_centers = self.__k
        clusterer = self.__clusterer
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_prot = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_prot = numpy.zeros((len(corpus_file_list), k_centers),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        word2prediction = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            word2prediction[term] = clusterer.predict(mat_terms[unorder_dict_index[term], :])
        ###############################################################    
        
        centroids = clusterer.cluster_centers_
        dist = DistanceMetric.get_metric('euclidean')
        print self.get_matrix()
        print "-------------------------------"
        print centroids
        matrix_docs_dist_prot = metrics.pairwise.rbf_kernel(self.get_matrix(), centroids)#dist.pairwise(self.get_matrix(), centroids) 
        print "=========================================="
        print matrix_docs_dist_prot
        
        corpus_bow = []    
        corpus_bow_prot = []
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                print "document: ", i
                ################################################################
                # SUPER SPEED 
                bow = []
                bow_prot = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        bow_prot += [(word2prediction[pal], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        ####### print "palabra: ", pal, "   ",unorder_dict_index[pal] 
                        #######print "La i: ", i
                        #######print clusterer.predict(mat_terms[unorder_dict_index[pal], :])
                        #######print "blablabla"
                        #######print matrix_docs_prot
                        # FIXED: bottle neck
                        # matrix_docs_prot[i, clusterer.predict(mat_terms[unorder_dict_index[pal], :])] += freq #/ tamDoc
                        matrix_docs_prot[i, word2prediction[pal]] += freq #/ tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
                corpus_bow_prot += [bow_prot]
            
        #Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]
        
        matrix_docs_prot = matrix_docs_prot * matrix_docs_dist_prot

        self._matrix = matrix_docs_prot
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of DOR representation. Time: ", str(t2-t1)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)        
        self.__clusterer =  joblib.load(cache_file + '_clusterer.pkl')
        return self.__clusterer
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)
        joblib.dump(value, cache_file + '_clusterer.pkl')         
        self.__clusterer=value
        
    def get_matrix(self):
        #super(FixedQuantizedTrainMatrixHolder, self).get_matrix()
        # return self._matrix
        if self._matrix is not None:
            return self._matrix
        else:
            return super(FixedQuantizedMatrixHolder, self).get_matrix() 

    def set_matrix(self, value):
        self._matrix = value
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
        
class FixedQuantizedTrainMatrixHolder(FixedQuantizedMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedQuantizedTrainMatrixHolder, self).__init__(matrix_holder, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedQuantizedTrainMatrixHolder, self).build_matrix()
        
        self.compute_prototypes(super(FixedQuantizedTrainMatrixHolder, self).get_matrix_terms())
        #print self.get_matrix_terms()
        
        self.build_matrix_quantized(super(FixedQuantizedTrainMatrixHolder, self).get_space(),
                              super(FixedQuantizedTrainMatrixHolder, self).get_space().virtual_classes_holder_train,
                              super(FixedQuantizedTrainMatrixHolder, self).get_space().corpus_file_list_train,
                              super(FixedQuantizedTrainMatrixHolder, self).get_matrix_terms())
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
    def save_train_data(self, space):
        super(FixedQuantizedTrainMatrixHolder, self).save_train_data(space)
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        joblib.dump(self.get_shared_resource(), cache_file + '_clusterer.pkl') 
    
    def load_train_data(self, space):
        train_matrix_holder = super(FixedQuantizedTrainMatrixHolder, self).load_train_data(space)
        train_matrix_holder = FixedQuantizedTrainMatrixHolder(train_matrix_holder, self.get_k_centers()) 
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        train_matrix_holder.set_shared_resource(joblib.load(cache_file + '_clusterer.pkl'))
    
        return train_matrix_holder
         
    
class FixedQuantizedTestMatrixHolder(FixedQuantizedMatrixHolder):
    
    def __init__(self, matrix_holder_object, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedQuantizedTestMatrixHolder, self).__init__(matrix_holder_object, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedQuantizedTestMatrixHolder, self).build_matrix()
        
        #self.compute_prototypes(super(FixedQuantizedTestMatrixHolder, self).get_matrix_terms())
        self.set_shared_resource(super(FixedQuantizedTestMatrixHolder, self).get_shared_resource())
        
        self.build_matrix_quantized(super(FixedQuantizedTestMatrixHolder, self).get_space(),
                              super(FixedQuantizedTestMatrixHolder, self).get_space().virtual_classes_holder_test,
                              super(FixedQuantizedTestMatrixHolder, self).get_space().corpus_file_list_test,
                              super(FixedQuantizedTestMatrixHolder, self).get_matrix_terms())

    def save_train_data(self, space):
        super(FixedQuantizedTestMatrixHolder, self).save_train_data(space)
    
    def load_train_data(self, space):
        return super(FixedQuantizedTestMatrixHolder, self).load_train_data(space)    
    
    
class FixedQuantizedTFIDFMatrixHolder(DecoratorMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedQuantizedTFIDFMatrixHolder, self).__init__(matrix_holder)
        self.__k = k
        #self.term_matrix = term_matrix
        self.__clusterer = None
        self._precomputed_dict=precomputed_dict
        
    def get_tfidf(self):
        return self.tfidf
    
    def get_id2word(self):
        return self.id2word    
    
    def set_tfidf(self, tfidf):
        self.tfidf = tfidf
    
    def set_id2word(self, id2word):
        self.id2word = id2word  
        
        
    def get_k_centers(self):
        return self.__k
    
    def set_k_centers(self, value):
        self.__k = value
        
    def compute_prototypes(self, matrix_terms):
        k= self.__k
        
        print "Begining clustering."
        if self._precomputed_dict == "NO_PRECOMPUTED":
            clusterer = KMeans(n_clusters=k, verbose=1, n_jobs=4)
            clusterer.fit(matrix_terms)
        else:
            cache_file = self._precomputed_dict        
            clusterer =  joblib.load(cache_file)
            
        print "End of clustering."
        
        self.set_shared_resource(clusterer)


    def build_matrix_quantized(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_terms):

        t1 = time.time()
        print "Starting BOW representation..."
        
        k_centers = self.__k
        clusterer = self.__clusterer
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_prot = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_prot = numpy.zeros((len(corpus_file_list), k_centers),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        word2prediction = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            word2prediction[term] = clusterer.predict(mat_terms[unorder_dict_index[term], :])
        ###############################################################    
        
        corpus_bow = []    
        corpus_bow_prot = []
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                print "document: ", i
                ################################################################
                # SUPER SPEED 
                bow = []
                bow_prot = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        bow_prot += [(word2prediction[pal][0], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        ####### print "palabra: ", pal, "   ",unorder_dict_index[pal] 
                        #######print "La i: ", i
                        #######print clusterer.predict(mat_terms[unorder_dict_index[pal], :])
                        #######print "blablabla"
                        #######print matrix_docs_prot
                        # FIXED: bottle neck
                        # matrix_docs_prot[i, clusterer.predict(mat_terms[unorder_dict_index[pal], :])] += freq #/ tamDoc
                        matrix_docs_prot[i, word2prediction[pal]] += freq #/ tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
                corpus_bow_prot += [bow_prot]
            
        Util.create_a_dir(space.space_path + "/tfidf_q")
        
        #print corpus_bow
        
        ccc=[]
        for d in corpus_bow_prot:
            nd={}
            for w, fq in d:
                
                if w in nd:
                    nd[w] += fq
                else:
                    nd[w] = fq
            
            ccc += [[(w, nd[w]) for w in nd.keys()]]
        
        corpus_bow_prot = ccc
                
                    
        corpora.MmCorpus.serialize(space.space_path + "/tfidf_q/" + space.id_space + "_corpus.mm", corpus_bow_prot)
        self.corpus_bow_prot = corpora.MmCorpus(space.space_path + "/tfidf_q/" + space.id_space + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        print corpus_bow_prot
        self._matrix = matrix_docs_prot
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of DOR representation. Time: ", str(t2-t1)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)        
        self.__clusterer =  joblib.load(cache_file + '_clusterer.pkl')
        return self.__clusterer
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)
        joblib.dump(value, cache_file + '_clusterer.pkl')         
        self.__clusterer=value
        
    def get_matrix(self):
        #super(FixedQuantizedTrainMatrixHolder, self).get_matrix()
        return self._matrix 

    def set_matrix(self, value):
        self._matrix = value
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
        
class FixedQuantizedTFIDFTrainMatrixHolder(FixedQuantizedTFIDFMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedQuantizedTFIDFTrainMatrixHolder, self).__init__(matrix_holder, k, precomputed_dict)
        #self.build_bowcorpus_id2word(self.space, 
        #                             self.space.virtual_classes_holder_train, 
        #                             self.space.corpus_file_list_train)
        
        #self.build_matrix_quantized(super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_space(),
        #                      super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_space().virtual_classes_holder_train,
        #                      super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_space().corpus_file_list_train,
        #                      super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_matrix_terms())
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedQuantizedTFIDFTrainMatrixHolder, self).build_matrix()
        
        self.compute_prototypes(super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_matrix_terms())
        #print self.get_matrix_terms()
        
        self.build_matrix_quantized(super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_space(),
                              super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_space().virtual_classes_holder_train,
                              super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_space().corpus_file_list_train,
                              super(FixedQuantizedTFIDFTrainMatrixHolder, self).get_matrix_terms())
        
        self.tfidf = models.TfidfModel(self.corpus_bow_prot) # step 1 -- initialize a model
        self.corpus_tfidf = self.tfidf[self.corpus_bow_prot]
        # self.lsa = models.LsiModel(self.corpus_tfidf, id2word=self.id2word, num_topics=dimensions) # run distributed LSA on documents
        # self.corpus_lsa = self.lsa[self.corpus_tfidf]   
        
        #print len(self.corpus_tfidf)
        #print  self.space._vocabulary
        #From sparse to dense
        matrix_documents_tfidf = numpy.zeros((len(self.corpus_tfidf), self.get_k_centers()),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_tfidf in self.corpus_tfidf:            
            for (index, contribution) in doc_tfidf:                
                matrix_documents_tfidf[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_tfidf
        #End of from sparse to dense
        
        
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
    def save_train_data(self, space):
        super(FixedQuantizedTFIDFTrainMatrixHolder, self).save_train_data(space)
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        joblib.dump(self.get_shared_resource(), cache_file + '_clusterer.pkl') 
                 
        #numpy.save(cache_file + "_mat_terms_concepts.npy", 
        #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
         
        id2word = self.get_id2word()            
        with open(space.space_path + "/tfidf_q/" + space.id_space + "_id2word.txt", 'w') as outfile:
            json.dump(id2word, outfile)  
                       
        tfidf = self.get_tfidf()
        tfidf.save(space.space_path + "/tfidf_q/" + space.id_space + "_model.tfidf")
    
    
    def load_train_data(self, space):
        train_matrix_holder = super(FixedQuantizedTFIDFTrainMatrixHolder, self).load_train_data(space)
        train_matrix_holder = FixedQuantizedTFIDFTrainMatrixHolder(train_matrix_holder, self.get_k_centers()) 
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        train_matrix_holder.set_shared_resource(joblib.load(cache_file + '_clusterer.pkl'))
         
        #tfidf_train_matrix_holder = FixedQuantizedTFIDFTrainMatrixHolder(space)
         
        with open(space.space_path + "/tfidf_q/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/tfidf_q/" + space.id_space + "_model.tfidf")      
        # lsa = models.LsiModel.load(space.space_path + "/tfidf/" + space.id_space + "_model.lsi")    
         
        train_matrix_holder.set_id2word(id2word)
        train_matrix_holder.set_tfidf(tfidf)
        # tfidf_train_matrix_holder.set_lsa(lsa)
     
        return train_matrix_holder
         
    
class FixedQuantizedTFIDFTestMatrixHolder(FixedQuantizedTFIDFMatrixHolder):
    
    def __init__(self, matrix_holder_object, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedQuantizedTFIDFTestMatrixHolder, self).__init__(matrix_holder_object, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedQuantizedTFIDFTestMatrixHolder, self).build_matrix()
        
        #self.compute_prototypes(super(FixedQuantizedTestMatrixHolder, self).get_matrix_terms())
        self.set_shared_resource(super(FixedQuantizedTFIDFTestMatrixHolder, self).get_shared_resource())
        
        ########################         This is the TRAINING MODEL TFIDF
        #with open(super(FixedQuantizedTFIDFTestMatrixHolder, self).get_space().space_path + "/tfidf_q/" + super(FixedQuantizedTFIDFTestMatrixHolder, self).get_space().id_space + "_id2word.txt", 'r') as infile:
        #        id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(super(FixedQuantizedTFIDFTestMatrixHolder, self).get_space().space_path + "/tfidf_q/" + super(FixedQuantizedTFIDFTestMatrixHolder, self).get_space().id_space + "_model.tfidf")      
        # lsa = models.LsiModel.load(space.space_path + "/tfidf/" + space.id_space + "_model.lsi")    
        
        # self.set_id2word(id2word)
        self.set_tfidf(tfidf)
        #######################         This is the TRAINING MODEL TFIDF
        
        self.build_matrix_quantized(super(FixedQuantizedTFIDFTestMatrixHolder, self).get_space(),
                              super(FixedQuantizedTFIDFTestMatrixHolder, self).get_space().virtual_classes_holder_test,
                              super(FixedQuantizedTFIDFTestMatrixHolder, self).get_space().corpus_file_list_test,
                              super(FixedQuantizedTFIDFTestMatrixHolder, self).get_matrix_terms())
        
        

        self.corpus_tfidf = self.tfidf[self.corpus_bow_prot]
        # self.corpus_lsa = self.lsa[self.corpus_tfidf]        
        
        #From sparse to dense
        matrix_documents_tfidf = numpy.zeros((len(self.corpus_tfidf), self.get_k_centers()),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_tfidf in self.corpus_tfidf:            
            for (index, contribution) in doc_tfidf:                
                matrix_documents_tfidf[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_tfidf
        
    def save_train_data(self, space):
        super(FixedQuantizedTFIDFTestMatrixHolder, self).save_train_data(space)
    
    def load_train_data(self, space): 
        return super(FixedQuantizedTFIDFTestMatrixHolder, self).load_train_data(space)  
       


class NormalizedProbsDecoratorMatrixHolder(DecoratorMatrixHolder):
    
    def __init__(self, matrix_holder):
        super(NormalizedProbsDecoratorMatrixHolder, self).__init__(matrix_holder)

    def normalize_probs(self, matrix):
        normalized_matrix = matrix
        print type(matrix)
        norma=normalized_matrix.sum(axis=1, dtype='float') # sum all columns
        print norma
        zeros=numpy.where( norma == 0)[0]
        norma[zeros]=0.00000001
        print norma
        #norma+=0.00000001
        elements = numpy.size(norma,0)
        normalized_matrix=normalized_matrix/norma.reshape(elements,1)
        return normalized_matrix
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
        
class NormalizedProbsDecoratorTrainMatrixHolder(NormalizedProbsDecoratorMatrixHolder):
    
    def __init__(self, matrix_holder):
        super(NormalizedProbsDecoratorTrainMatrixHolder, self).__init__(matrix_holder)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(NormalizedProbsDecoratorTrainMatrixHolder, self).build_matrix()
        self.set_matrix(
                        self.normalize_probs(
                                             super(NormalizedProbsDecoratorTrainMatrixHolder, self).get_matrix()
                                             )
                        )
        
    def save_train_data(self, space):
        super(NormalizedProbsDecoratorTrainMatrixHolder, self).save_train_data(space)
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        numpy.save(cache_file + '_norm_probs_matrix.mat', self.get_matrix()) 
    
    def load_train_data(self, space):
        train_matrix_holder = super(NormalizedProbsDecoratorTrainMatrixHolder, self).load_train_data(space)
        train_matrix_holder = NormalizedProbsDecoratorTrainMatrixHolder(train_matrix_holder) 
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        train_matrix_holder.set_matrix(numpy.load(cache_file + '_norm_probs_matrix.mat.npy'))
    
        return train_matrix_holder
         
    
class NormalizedProbsDecoratorTestMatrixHolder(NormalizedProbsDecoratorMatrixHolder):
    
    def __init__(self, matrix_holder_object):
        super(NormalizedProbsDecoratorTestMatrixHolder, self).__init__(matrix_holder_object)
        
    def build_matrix(self):
        super(NormalizedProbsDecoratorTestMatrixHolder, self).build_matrix()
        self.set_matrix(
                        self.normalize_probs(
                                             super(NormalizedProbsDecoratorTestMatrixHolder, self).get_matrix()
                                             )
                        )

    def save_train_data(self, space):
        super(NormalizedProbsDecoratorTestMatrixHolder, self).save_train_data(space)
    
    def load_train_data(self, space):
        return super(NormalizedProbsDecoratorTestMatrixHolder, self).load_train_data(space)
                
                
class FixedDistances2CTMatrixHolder(DecoratorMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED", distance="euclidean"):
        super(FixedDistances2CTMatrixHolder, self).__init__(matrix_holder)
        self.__k = k
        #self.term_matrix = term_matrix
        self.__clusterer = None
        self._precomputed_dict=precomputed_dict
        self._matrix = None
        self._distance=distance
        
    def get_k_centers(self):
        return self.__k
    
    def set_k_centers(self, value):
        self.__k = value
        
    def compute_prototypes(self, matrix_terms):
        k= self.__k
        
        print "Begining clustering."
        if self._precomputed_dict == "NO_PRECOMPUTED":
            clusterer = KMeans(n_clusters=k, verbose=1, n_jobs=4)
            clusterer.fit(matrix_terms)
        else:
            print "The clusterer was precomputed  :)"
            cache_file = self._precomputed_dict        
            clusterer =  joblib.load(cache_file)
            
        print "End of clustering."
        
        self.set_shared_resource(clusterer)

    def build_matrix_quantized(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_terms):

        t1 = time.time()
        print "Starting FixedDistances2CTMatrixHolder representation..."
        
        k_centers = self.__k
        clusterer = self.__clusterer
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_prot = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_prot = numpy.zeros((len(corpus_file_list), k_centers),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        #word2prediction = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            #word2prediction[term] = clusterer.predict(mat_terms[unorder_dict_index[term], :])
        ###############################################################    
        
        
        centroids = clusterer.cluster_centers_
        # dist = DistanceMetric.get_metric('euclidean')
        # print self.get_matrix()
        # print "-------------------------------"
        # print centroids
        print "Computing distances/similarity."
        
        matrix_docs_prot = None
        if self._distance == 'euclidean':
            matrix_docs_prot = metrics.pairwise.euclidean_distances(self.get_matrix(), centroids)#dist.pairwise(self.get_matrix(), centroids) 
            #print "=========================================="
            #print matrix_docs_prot
        elif self._distance == 'cosine_distance':
            matrix_docs_prot = metrics.pairwise.cosine_distances(self.get_matrix(), centroids)#dist.pairwise(self.get_matrix(), centroids)
        elif self._distance == 'cosine_similarity':
            matrix_docs_prot = metrics.pairwise.cosine_similarity(self.get_matrix(), centroids)#dist.pairwise(self.get_matrix(), centroids)
        elif self._distance == 'rbf':
            matrix_docs_prot = metrics.pairwise.rbf_kernel(self.get_matrix(), centroids)#dist.pairwise(self.get_matrix(), centroids)
        elif self._distance == 'lineal':
            matrix_docs_prot = metrics.pairwise.linear_kernel(self.get_matrix(), centroids)#dist.pairwise(self.get_matrix(), centroids)
        else:
            print "No distance or similarty measure was indicated."
            sys.exit()
        print "End of computing distances/similarity."
                
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                # EXTRACODE FOR PRINT THE CORPUS###################################
#                 shelf = shelve.open("./exp_validacion1024_tc/1_TermSplit_test.dat", protocol=2)
#                 #print shelf.keys()
#                 
#                 
#                 target_tokens = []
#                 for tok in tokens:
#                     target_tokens += [str(word2prediction[tok][0])]
#                     #print target_tokens[i_tok]
#                     #(NDARRAY)print type(word2prediction[tok])
#                     #print word2prediction[tok]                    
#                 
#                 shelf[arch.encode('utf8')] = target_tokens
#                 
#                 shelf.close()
                # EXTRACODE FOR PRINT THE CORPUS###################################
                     
                #print "document: ", i
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        ####### print "palabra: ", pal, "   ",unorder_dict_index[pal] 
                        #######print "La i: ", i
                        #######print clusterer.predict(mat_terms[unorder_dict_index[pal], :])
                        #######print "blablabla"
                        #######print matrix_docs_prot
                        # FIXED: bottle neck
                        # matrix_docs_prot[i, clusterer.predict(mat_terms[unorder_dict_index[pal], :])] += freq #/ tamDoc
                        # matrix_docs_prot[i, word2prediction[pal]] += freq #/ tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
            
        #Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_prot
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of FixedDistances2CTMatrixHolder representation. Time: ", str(t2-t1)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)        
        self.__clusterer =  joblib.load(cache_file + '_clusterer.pkl')
        return self.__clusterer
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)
        joblib.dump(value, cache_file + '_clusterer.pkl')         
        self.__clusterer=value
        
    def get_matrix(self):
        #super(FixedQuantizedTrainMatrixHolder, self).get_matrix()
        if self._matrix is not None:
            return self._matrix
        else:
            return super(FixedDistances2CTMatrixHolder, self).get_matrix()
            
            
        

    def set_matrix(self, value):
        self._matrix = value
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
        
class FixedDistances2CTTrainMatrixHolder(FixedDistances2CTMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED", distance="euclidean"):
        super(FixedDistances2CTTrainMatrixHolder, self).__init__(matrix_holder, k, precomputed_dict, distance)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedDistances2CTTrainMatrixHolder, self).build_matrix()
        
        self.compute_prototypes(super(FixedDistances2CTTrainMatrixHolder, self).get_matrix_terms())
        ####NOT NECESARY self.set_matrix(super(FixedDistances2CTTrainMatrixHolder, self).get_matrix())
        #print self.get_matrix_terms()
        
        self.build_matrix_quantized(super(FixedDistances2CTTrainMatrixHolder, self).get_space(),
                              super(FixedDistances2CTTrainMatrixHolder, self).get_space().virtual_classes_holder_train,
                              super(FixedDistances2CTTrainMatrixHolder, self).get_space().corpus_file_list_train,
                              super(FixedDistances2CTTrainMatrixHolder, self).get_matrix_terms())
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
    def save_train_data(self, space):
        super(FixedDistances2CTTrainMatrixHolder, self).save_train_data(space)
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        joblib.dump(self.get_shared_resource(), cache_file + '_clusterer.pkl') 
    
    def load_train_data(self, space):
        train_matrix_holder = super(FixedDistances2CTTrainMatrixHolder, self).load_train_data(space)
        train_matrix_holder = FixedDistances2CTTrainMatrixHolder(train_matrix_holder, self.get_k_centers()) 
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        train_matrix_holder.set_shared_resource(joblib.load(cache_file + '_clusterer.pkl'))
    
        return train_matrix_holder
         
    
class FixedDistances2CTTestMatrixHolder(FixedDistances2CTMatrixHolder):
    
    def __init__(self, matrix_holder_object, k=300, precomputed_dict="NO_PRECOMPUTED", distance="euclidean"):
        super(FixedDistances2CTTestMatrixHolder, self).__init__(matrix_holder_object, k, precomputed_dict, distance)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedDistances2CTTestMatrixHolder, self).build_matrix()
        
        #self.compute_prototypes(super(FixedQuantizedTestMatrixHolder, self).get_matrix_terms())
        self.set_shared_resource(super(FixedDistances2CTTestMatrixHolder, self).get_shared_resource())
        
        ####NOT NECESARY self.set_matrix(super(FixedDistances2CTTestMatrixHolder, self).get_matrix())
        
        self.build_matrix_quantized(super(FixedDistances2CTTestMatrixHolder, self).get_space(),
                              super(FixedDistances2CTTestMatrixHolder, self).get_space().virtual_classes_holder_test,
                              super(FixedDistances2CTTestMatrixHolder, self).get_space().corpus_file_list_test,
                              super(FixedDistances2CTTestMatrixHolder, self).get_matrix_terms())

    def save_train_data(self, space):
        super(FixedDistances2CTTestMatrixHolder, self).save_train_data(space)
    
    def load_train_data(self, space):
        return super(FixedDistances2CTTestMatrixHolder, self).load_train_data(space)
    
    
class FixedDistances2CDMatrixHolder(DecoratorMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedDistances2CDMatrixHolder, self).__init__(matrix_holder)
        self.__k = k
        #self.term_matrix = term_matrix
        self.__clusterer = None
        self._precomputed_dict=precomputed_dict
        self._matrix = None
        
    def get_k_centers(self):
        return self.__k
    
    def set_k_centers(self, value):
        self.__k = value
        
    def compute_prototypes(self, matrix_terms):
        k= self.__k
        
        print "Begining clustering."
        if self._precomputed_dict == "NO_PRECOMPUTED":
            clusterer = KMeans(n_clusters=k, verbose=1, n_jobs=4)
            clusterer.fit(matrix_terms)
        else:
            cache_file = self._precomputed_dict        
            clusterer =  joblib.load(cache_file)
            
        print "End of clustering."
        
        self.set_shared_resource(clusterer)

    def build_matrix_quantized(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_terms):

        t1 = time.time()
        print "Starting BOW representation..."
        
        k_centers = self.__k
        clusterer = self.__clusterer
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_prot = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_prot = numpy.zeros((len(corpus_file_list), k_centers),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        word2prediction = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            word2prediction[term] = clusterer.predict(mat_terms[unorder_dict_index[term], :])
        ###############################################################    
        
        
        centroids = clusterer.cluster_centers_
        dist = DistanceMetric.get_metric('euclidean')
        print self.get_matrix()
        print "-------------------------------"
        print centroids
        matrix_docs_prot = dist.pairwise(self.get_matrix(), centroids) 
        print "=========================================="
        print matrix_docs_prot
                
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                print "document: ", i
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        ####### print "palabra: ", pal, "   ",unorder_dict_index[pal] 
                        #######print "La i: ", i
                        #######print clusterer.predict(mat_terms[unorder_dict_index[pal], :])
                        #######print "blablabla"
                        #######print matrix_docs_prot
                        # FIXED: bottle neck
                        # matrix_docs_prot[i, clusterer.predict(mat_terms[unorder_dict_index[pal], :])] += freq #/ tamDoc
                        # matrix_docs_prot[i, word2prediction[pal]] += freq #/ tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
            
        #Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_prot
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of DOR representation. Time: ", str(t2-t1)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)        
        self.__clusterer =  joblib.load(cache_file + '_clusterer.pkl')
        return self.__clusterer
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)
        joblib.dump(value, cache_file + '_clusterer.pkl')         
        self.__clusterer=value
        
    def get_matrix(self):
        #super(FixedQuantizedTrainMatrixHolder, self).get_matrix()
        if self._matrix is not None:
            return self._matrix
        else:
            return super(FixedDistances2CDMatrixHolder, self).get_matrix()
            
            
        

    def set_matrix(self, value):
        self._matrix = value
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
        
class FixedDistances2CDTrainMatrixHolder(FixedDistances2CDMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedDistances2CDTrainMatrixHolder, self).__init__(matrix_holder, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedDistances2CDTrainMatrixHolder, self).build_matrix()
        
        self.compute_prototypes(super(FixedDistances2CDTrainMatrixHolder, self).get_matrix())
        #self.set_matrix(super(FixedDistances2CDTrainMatrixHolder, self).get_matrix())
        #print self.get_matrix_terms()
        
        self.build_matrix_quantized(super(FixedDistances2CDTrainMatrixHolder, self).get_space(),
                              super(FixedDistances2CDTrainMatrixHolder, self).get_space().virtual_classes_holder_train,
                              super(FixedDistances2CDTrainMatrixHolder, self).get_space().corpus_file_list_train,
                              super(FixedDistances2CDTrainMatrixHolder, self).get_matrix_terms())
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
    def save_train_data(self, space):
        super(FixedDistances2CDTrainMatrixHolder, self).save_train_data(space)
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        joblib.dump(self.get_shared_resource(), cache_file + '_clusterer.pkl') 
    
    def load_train_data(self, space):
        train_matrix_holder = super(FixedDistances2CDTrainMatrixHolder, self).load_train_data(space)
        train_matrix_holder = FixedDistances2CDTrainMatrixHolder(train_matrix_holder, self.get_k_centers()) 
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        train_matrix_holder.set_shared_resource(joblib.load(cache_file + '_clusterer.pkl'))
    
        return train_matrix_holder
         
    
class FixedDistances2CDTestMatrixHolder(FixedDistances2CDMatrixHolder):
    
    def __init__(self, matrix_holder_object, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(FixedDistances2CDTestMatrixHolder, self).__init__(matrix_holder_object, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(FixedDistances2CDTestMatrixHolder, self).build_matrix()
        
        #self.compute_prototypes(super(FixedQuantizedTestMatrixHolder, self).get_matrix_terms())
        self.set_shared_resource(super(FixedDistances2CDTestMatrixHolder, self).get_shared_resource())
        #self.set_matrix(super(FixedDistances2CDTestMatrixHolder, self).get_matrix())
        
        self.build_matrix_quantized(super(FixedDistances2CDTestMatrixHolder, self).get_space(),
                              super(FixedDistances2CDTestMatrixHolder, self).get_space().virtual_classes_holder_test,
                              super(FixedDistances2CDTestMatrixHolder, self).get_space().corpus_file_list_test,
                              super(FixedDistances2CDTestMatrixHolder, self).get_matrix_terms())

    def save_train_data(self, space):
        super(FixedDistances2CDTestMatrixHolder, self).save_train_data(space)
    
    def load_train_data(self, space):
        return super(FixedDistances2CDTestMatrixHolder, self).load_train_data(space)
    
        
              
class Distances2CTMatrixHolder(DecoratorMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(Distances2CTMatrixHolder, self).__init__(matrix_holder)
        self.__k = k
        #self.term_matrix = term_matrix
        self.__clusterer = None
        self._precomputed_dict=precomputed_dict
        self._matrix = None
        
    def get_k_centers(self):
        return self.__k
    
    def set_k_centers(self, value):
        self.__k = value
        
    def compute_prototypes(self, matrix_terms):
        k= self.__k
        
        print "Begining clustering."
        if self._precomputed_dict == "NO_PRECOMPUTED":
            clusterer = Util.perform_EM(matrix_terms, 'tied', 100)
            #clusterer = KMeans(n_clusters=k, verbose=1, n_jobs=4)
            #clusterer.fit(matrix_terms)
        else:
            cache_file = self._precomputed_dict        
            clusterer =  joblib.load(cache_file)
                  
        print "End of clustering."
        
        self.set_shared_resource(clusterer)

    def build_matrix_quantized(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_terms):

        t1 = time.time()
        print "Starting BOW representation..."
        
        k_centers = self.__k
        clusterer = self.__clusterer
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_prot = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_prot = numpy.zeros((len(corpus_file_list), k_centers),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        word2prediction = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            #print "XXXXXXXXXXXXXXXXXXXXXXXXX"
            #print clusterer.means_
            #print mat_terms[unorder_dict_index[term], :]
            #print "XXXXXXXXXXXXXXXXXXXXXXXXX"
            word2prediction[term] = clusterer.predict([mat_terms[unorder_dict_index[term], :]])
        ###############################################################    
        
        
        centroids = clusterer.means_
        dist = DistanceMetric.get_metric('euclidean')
        print self.get_matrix()
        #print "-------------------------------"
        print centroids
        matrix_docs_prot = dist.pairwise(self.get_matrix(), centroids) 
        #print "=========================================="
        print matrix_docs_prot
                
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                print "document: ", i
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        ####### print "palabra: ", pal, "   ",unorder_dict_index[pal] 
                        #######print "La i: ", i
                        #######print clusterer.predict(mat_terms[unorder_dict_index[pal], :])
                        #######print "blablabla"
                        #######print matrix_docs_prot
                        # FIXED: bottle neck
                        # matrix_docs_prot[i, clusterer.predict(mat_terms[unorder_dict_index[pal], :])] += freq #/ tamDoc
                        # matrix_docs_prot[i, word2prediction[pal]] += freq #/ tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
            
        #Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_prot
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of DOR representation. Time: ", str(t2-t1)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)        
        self.__clusterer =  joblib.load(cache_file + '_clusterer.pkl')
        return self.__clusterer
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)
        joblib.dump(value, cache_file + '_clusterer.pkl')         
        self.__clusterer=value
        
    def get_matrix(self):
        #super(FixedQuantizedTrainMatrixHolder, self).get_matrix()
        if self._matrix is not None:
            return self._matrix
        else:
            return super(Distances2CTMatrixHolder, self).get_matrix()
            
            
        

    def set_matrix(self, value):
        self._matrix = value
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
        
class Distances2CTTrainMatrixHolder(Distances2CTMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(Distances2CTTrainMatrixHolder, self).__init__(matrix_holder, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(Distances2CTTrainMatrixHolder, self).build_matrix()
        
        self.compute_prototypes(super(Distances2CTTrainMatrixHolder, self).get_matrix_terms())
        ####NOT NECESARY self.set_matrix(super(FixedDistances2CTTrainMatrixHolder, self).get_matrix())
        #print self.get_matrix_terms()
        
        self.build_matrix_quantized(super(Distances2CTTrainMatrixHolder, self).get_space(),
                              super(Distances2CTTrainMatrixHolder, self).get_space().virtual_classes_holder_train,
                              super(Distances2CTTrainMatrixHolder, self).get_space().corpus_file_list_train,
                              super(Distances2CTTrainMatrixHolder, self).get_matrix_terms())
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
    def save_train_data(self, space):
        super(Distances2CTTrainMatrixHolder, self).save_train_data(space)
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        joblib.dump(self.get_shared_resource(), cache_file + '_clusterer.pkl') 
    
    def load_train_data(self, space):
        train_matrix_holder = super(Distances2CTTrainMatrixHolder, self).load_train_data(space)
        train_matrix_holder = Distances2CTTrainMatrixHolder(train_matrix_holder, self.get_k_centers()) 
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        train_matrix_holder.set_shared_resource(joblib.load(cache_file + '_clusterer.pkl'))
    
        return train_matrix_holder
         
    
class Distances2CTTestMatrixHolder(Distances2CTMatrixHolder):
    
    def __init__(self, matrix_holder_object, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(Distances2CTTestMatrixHolder, self).__init__(matrix_holder_object, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(Distances2CTTestMatrixHolder, self).build_matrix()
        
        #self.compute_prototypes(super(FixedQuantizedTestMatrixHolder, self).get_matrix_terms())
        self.set_shared_resource(super(Distances2CTTestMatrixHolder, self).get_shared_resource())
        
        ####NOT NECESARY self.set_matrix(super(FixedDistances2CTTestMatrixHolder, self).get_matrix())
        
        self.build_matrix_quantized(super(Distances2CTTestMatrixHolder, self).get_space(),
                              super(Distances2CTTestMatrixHolder, self).get_space().virtual_classes_holder_test,
                              super(Distances2CTTestMatrixHolder, self).get_space().corpus_file_list_test,
                              super(Distances2CTTestMatrixHolder, self).get_matrix_terms())

    def save_train_data(self, space):
        super(Distances2CTTestMatrixHolder, self).save_train_data(space)
    
    def load_train_data(self, space):
        return super(Distances2CTTestMatrixHolder, self).load_train_data(space)
    
    
class Distances2CDMatrixHolder(DecoratorMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(Distances2CDMatrixHolder, self).__init__(matrix_holder)
        self.__k = k
        #self.term_matrix = term_matrix
        self.__clusterer = None
        self._precomputed_dict=precomputed_dict
        self._matrix = None
        
    def get_k_centers(self):
        return self.__k
    
    def set_k_centers(self, value):
        self.__k = value
        
    def compute_prototypes(self, matrix_terms):
        k= self.__k
        
        print "Begining clustering."
        if self._precomputed_dict == "NO_PRECOMPUTED":
            clusterer = Util.perform_EM(matrix_terms, 'tied', 100)
            #clusterer = KMeans(n_clusters=k, verbose=1, n_jobs=4)
            #clusterer.fit(matrix_terms)
        else:
            cache_file = self._precomputed_dict        
            clusterer =  joblib.load(cache_file)
                  
        print "End of clustering."
        
        self.set_shared_resource(clusterer)

    def build_matrix_quantized(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_terms):

        t1 = time.time()
        print "Starting BOW representation..."
        
        k_centers = self.__k
        clusterer = self.__clusterer
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_prot = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_prot = numpy.zeros((len(corpus_file_list), k_centers),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        word2prediction = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            #print "XXXXXXXXXXXXXXXXXXXXXXXXX"
            #print clusterer.means_
            #print mat_terms[unorder_dict_index[term], :]
            #print "XXXXXXXXXXXXXXXXXXXXXXXXX"
            word2prediction[term] = clusterer.predict([mat_terms[unorder_dict_index[term], :]])
        ###############################################################    
        
        
        centroids = clusterer.means_
        dist = DistanceMetric.get_metric('euclidean')
        print self.get_matrix()
        #print "-------------------------------"
        print centroids
        matrix_docs_prot = dist.pairwise(self.get_matrix(), centroids) 
        #print "=========================================="
        print matrix_docs_prot
                
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                print "document: ", i
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        ####### print "palabra: ", pal, "   ",unorder_dict_index[pal] 
                        #######print "La i: ", i
                        #######print clusterer.predict(mat_terms[unorder_dict_index[pal], :])
                        #######print "blablabla"
                        #######print matrix_docs_prot
                        # FIXED: bottle neck
                        # matrix_docs_prot[i, clusterer.predict(mat_terms[unorder_dict_index[pal], :])] += freq #/ tamDoc
                        # matrix_docs_prot[i, word2prediction[pal]] += freq #/ tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
            
        #Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_prot
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of DOR representation. Time: ", str(t2-t1)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)        
        self.__clusterer =  joblib.load(cache_file + '_clusterer.pkl')
        return self.__clusterer
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        cache_file = "%s/%s" % (self.get_space().space_path, self.get_space().id_space)
        joblib.dump(value, cache_file + '_clusterer.pkl')         
        self.__clusterer=value
        
    def get_matrix(self):
        #super(FixedQuantizedTrainMatrixHolder, self).get_matrix()
        if self._matrix is not None:
            return self._matrix
        else:
            return super(Distances2CDMatrixHolder, self).get_matrix()
            
            
        

    def set_matrix(self, value):
        self._matrix = value
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
        
class Distances2CDTrainMatrixHolder(Distances2CTMatrixHolder):
    
    def __init__(self, matrix_holder, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(Distances2CDTrainMatrixHolder, self).__init__(matrix_holder, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(Distances2CDTrainMatrixHolder, self).build_matrix()
        
        self.compute_prototypes(super(Distances2CDTrainMatrixHolder, self).get_matrix())
        ####NOT NECESARY self.set_matrix(super(FixedDistances2CTTrainMatrixHolder, self).get_matrix())
        #print self.get_matrix_terms()
        
        self.build_matrix_quantized(super(Distances2CDTrainMatrixHolder, self).get_space(),
                              super(Distances2CDTrainMatrixHolder, self).get_space().virtual_classes_holder_train,
                              super(Distances2CDTrainMatrixHolder, self).get_space().corpus_file_list_train,
                              super(Distances2CDTrainMatrixHolder, self).get_matrix_terms())
        
    #def build_matrix(self):
        # self.__matrix_holder_object.build_matrix()
        #pass
        
    def save_train_data(self, space):
        super(Distances2CDTrainMatrixHolder, self).save_train_data(space)
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        joblib.dump(self.get_shared_resource(), cache_file + '_clusterer.pkl') 
    
    def load_train_data(self, space):
        train_matrix_holder = super(Distances2CDTrainMatrixHolder, self).load_train_data(space)
        train_matrix_holder = Distances2CDTrainMatrixHolder(train_matrix_holder, self.get_k_centers()) 
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        train_matrix_holder.set_shared_resource(joblib.load(cache_file + '_clusterer.pkl'))
    
        return train_matrix_holder
         
    
class Distances2CDTestMatrixHolder(Distances2CTMatrixHolder):
    
    def __init__(self, matrix_holder_object, k=300, precomputed_dict="NO_PRECOMPUTED"):
        super(Distances2CDTestMatrixHolder, self).__init__(matrix_holder_object, k, precomputed_dict)
        
    def build_matrix(self):
        #print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        super(Distances2CDTestMatrixHolder, self).build_matrix()
        
        #self.compute_prototypes(super(FixedQuantizedTestMatrixHolder, self).get_matrix_terms())
        self.set_shared_resource(super(Distances2CDTestMatrixHolder, self).get_shared_resource())
        
        ####NOT NECESARY self.set_matrix(super(FixedDistances2CTTestMatrixHolder, self).get_matrix())
        
        self.build_matrix_quantized(super(Distances2CDTestMatrixHolder, self).get_space(),
                              super(Distances2CDTestMatrixHolder, self).get_space().virtual_classes_holder_test,
                              super(Distances2CDTestMatrixHolder, self).get_space().corpus_file_list_test,
                              super(Distances2CDTestMatrixHolder, self).get_matrix_terms())

    def save_train_data(self, space):
        super(Distances2CDTestMatrixHolder, self).save_train_data(space)
    
    def load_train_data(self, space):
        return super(Distances2CDTestMatrixHolder, self).load_train_data(space)
    
    
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
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[author].dic_file_fd[arch]
                tam_doc = len(tokens)
                print k,":",tam_doc

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
                for pal in docActualFd.keys_sorted():
                    
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
        for e in docActualFd.keys_sorted():
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
            print "cat: ",i
            
            total_terms_in_class = 0
            for arch in archivos:
                #print arch

                tokens = space.virtual_classes_holder_train[author].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #space.virtual_classes_holder_train[author].dic_file_fd[arch]
                tamDoc = len(tokens)
                total_terms_in_class += tamDoc
                
                ################################################################
                # SUPER SPEED 
                
                for pal in docActualFd.keys_sorted():
                    
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
            
        new_fdist = FreqDistExt()
        for term in new_vocabulary:
            new_fdist[term] = space._fdist[term]
        
        if debug:
            f_debug.write("after_fdist: " + str(new_fdist) + "\n")
            f_debug.write("final_vocab: " + str(new_fdist.keys_sorted()) + "\n")
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
        
    # ----------------------------------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._matrix_concepts_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_concepts_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass       
        
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            numpy.save(cache_file + "_mat_terms_concepts.npy", 
                       self.get_matrix_terms_concepts())
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
        else:
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        csa_train_matrix_holder = CSATrainMatrixHolder(space)        
        csa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))        
        csa_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        csa_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        csa_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return csa_train_matrix_holder   


class CSATestMatrixHolder(CSAMatrixHolder):

    def __init__(self, space, train_matrix_holder=None, matrix_concepts_terms=None):
        super(CSATestMatrixHolder, self).__init__(space)
        self._matrix_concepts_terms = numpy.transpose(train_matrix_holder.get_matrix_terms())
        #self._matrix_concepts_terms = matrix_concepts_terms
        #self.build_matrix()

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
        
    # ------------------------------------------------------------------------         
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._matrix_concepts_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_concepts_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass


class CSA2MatrixHolder(MatrixHolder):

    def __init__(self, space):
        super(CSA2MatrixHolder, self).__init__()
        self.space = space
        self._ordered_new_labels_set = None
        
#         if ("KMEANS" == space.kwargs_space["subclassing"]["clusterer"]):        
#             self.clusterer = self.space.kwargs_space['subclassing']["clusterer"]
#             self.set_dimensions_soa2(len(space.categories) * self.space.kwargs_space['subclassing']["k"])
#             print self.clusterer
#             print self.get_dimensions_soa2()
#         else:
#             self.clusterer = "EM"
#             print self.clusterer
            
    def build_matrix_documents_concepts(self,
                                        space,
                                        virtual_classes_holder,
                                        corpus_file_list,
                                        matrix_concepts_terms,
                                        SOA2=False):
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

        if SOA2:
            dimensions = self.get_dimensions_soa2()
            print "PERFORMING SECOND STEP OF SOA."
            # print matrix_concepts_terms
        else:
            dimensions = len(space.categories)
            
        matrix_concepts_docs = numpy.zeros((dimensions, len(corpus_file_list)),
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
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[author].dic_file_fd[arch]
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
                for pal in docActualFd.keys_sorted():
                    
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
                 
    def compute_prototypes(self, matrix_terms):
        k= self.__k
        
        print "Begining clustering."
        if self._precomputed_dict == "NO_PRECOMPUTED":
            clusterer = KMeans(n_clusters=k, verbose=1, n_jobs=4)
            clusterer.fit(matrix_terms)
        else:
            cache_file = self._precomputed_dict        
            clusterer =  joblib.load(cache_file)
            
        print "End of clustering."
        
        self.set_shared_resource(clusterer)

    def weight_matrix(self, space, elemAutor, tokens, docActualFd, matrix_concepts_docs, numAut, k, mat_concepts_term):
        """
        La K es un contador de documentos definida en la funcion padre
        """

#        docActualFd=FreqDist(tokens)
        tamDoc=len(tokens)
        freqrank={}
        for e in docActualFd.keys_sorted():
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
            
    def set_dimensions_soa2(self, k):
        self.__dimensions = k
        
    def get_dimensions_soa2(self):
        return self.__dimensions
    
    def get_ordered_new_labels_set(self):
        return self._ordered_new_labels_set
    
    def set_ordered_new_labels_set(self, value):
        self._ordered_new_labels_set = value
        
    def get_instance_subcategories(self):
        return self._instance_subcategories
    
    def set_instance_subcategories(self, value):
        self._instance_subcategories = value
    


class CSA2TrainMatrixHolder(CSA2MatrixHolder):

    def __init__(self, space):
        super(CSA2TrainMatrixHolder, self).__init__(space)
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
                docActualFd = FreqDistExt(tokens) #space.virtual_classes_holder_train[author].dic_file_fd[arch]
                tamDoc = len(tokens)
                total_terms_in_class += tamDoc
                
                ################################################################
                # SUPER SPEED 
                
                for pal in docActualFd.keys_sorted():
                    
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

        self._matrix_concepts_terms0 = matrix_concepts_terms

        t2 = time.time()
        print "End CSA matrix concepts-terms. Time: ", str(t2 - t1)


    def build_matrix_concepts_terms2(self,space):
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
        
        
        data_training={}
        sc_data_training = {}
        total_new_labels = []
        set_of_new_labels= [] 
        
        #ordered_new_labels_set = []
        
        pivot = 0
        for category in space.categories:
            #print "TheCAT:"+category
            files = space.virtual_classes_holder_train[category].cat_file_list
            submatrix_concepts_docs = numpy.zeros((len(space.categories), len(files)),
                                           dtype=numpy.float64)
            #k = 0
            #for file in files:
            #    print "SHAPE: ",self._matrix.shape
            #    
            #    submatrix_concepts_docs[:, k] = self._matrix[:, range(100)]
            #    k += 1
                
            print "PIVOTE:", pivot
            submatrix_concepts_docs = self._matrix[:, range(pivot, pivot + len(files))]
            pivot = pivot + len(files)
           
                
            print submatrix_concepts_docs
                
            if "KMEANS" == space.kwargs_space["subclassing"]["clusterer"]:        
                clusterer = KMeans(n_clusters=space.kwargs_space['subclassing']["k"], 
                                   verbose=1, 
                                   n_jobs=space.kwargs_space["subclassing"]["processors"])
                            
                target_mat=numpy.transpose(submatrix_concepts_docs)
                clusterer.fit(target_mat)
                good_clusterer = clusterer
                print "NUMBER OF COMPONENTS: " + str(good_clusterer.n_clusters)
                set_of_new_labels += [category + "SCK172435EW" + str(subgroup) 
                                      for subgroup in range(good_clusterer.n_clusters)]
                
            if "EM" == space.kwargs_space["subclassing"]["clusterer"]:
                
                print "BEGINING EM ..."
                new_score=None
                best_score=None
                #best_score = numpy.infty #0.0
                good_clusterer = None
                for it in range(space.kwargs_space['subclassing']["k"])[1:]:
                    #print "ITERATION: " + str(it)
                    
                    clusterer = mixture.GMM(n_components=it, 
                                            covariance_type=space.kwargs_space['subclassing']["covariance_type"],
                                            n_iter=100,
                                            n_init=10,
                                            thresh=.000001,
                                            min_covar=.000001)
                    target_mat=numpy.transpose(submatrix_concepts_docs)
                    # clusterer.fit(target_mat[1:50, :])
                    clusterer.fit(target_mat)
                    
                    #print clusterer.predict_proba(target_mat)
                    #new_score = numpy.sum(numpy.amax(clusterer.predict_proba(target_mat), axis=1))
                    
                    the_log_likelihoods, the_resposibilities = clusterer.score_samples(target_mat)
                    new_score = the_log_likelihoods.mean()
                    
                    print "SC: ", new_score
                    if best_score is not None:
                        if numpy.abs(new_score - best_score) < .0001:
                            break
                        else:
                            if new_score < best_score:
                                best_score = new_score
                                good_clusterer = clusterer
                            else:
                                break
                    else:
                        best_score=new_score
                        good_clusterer = clusterer
                        
                    
                    print best_score
                    print good_clusterer
                
                print "NUMBER OF COMPONENTS: " + str(good_clusterer.n_components)
                set_of_new_labels += [category + "SCK172435EW" + str(subgroup) 
                                      for subgroup in range(good_clusterer.n_components)]
                    
            instance_subcategories = good_clusterer.predict(target_mat)
            print instance_subcategories
            print "CLUSTERING FINISHED..."
            
            data_training[category]= zip([category] * len(files), instance_subcategories, files)
            
            new_labels=[]
            new_labels += [new_info[0] + "SCK172435EW" + str(new_info[1]) 
                           for new_info in data_training[category]]
            
            for new_label, the_file in zip(new_labels, files):
                if new_label in sc_data_training:
                    sc_data_training[new_label] += [the_file]
                    #print new_label,the_file
                else:
                    sc_data_training[new_label] = [the_file]
                    #ordered_new_labels_set += [new_label]
                    #print new_label,the_file
                    
            total_new_labels += new_labels
            
        for ee in set_of_new_labels:
            if ee not in sc_data_training:
                sc_data_training[ee]=[]
        
        self.set_dimensions_soa2(len(set_of_new_labels))
        print self.get_dimensions_soa2()
            
        

        matrix_concepts_terms = numpy.zeros((len(set_of_new_labels), len(space._vocabulary)),
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
        i=0
        for author in set_of_new_labels:
            print author
            #print sc_data_training[author]
            archivos = sc_data_training[author]
            #print archivos
            total_terms_in_class = 0
            for arch in archivos:
                #print arch
                #print virtual_classes_holder[author.split("SCK172435EW")[0]].dic_file_tokens.items()
                tokens = space.virtual_classes_holder_train[author.split("SCK172435EW")[0]].dic_file_tokens[arch]
#                 
#         i = 0
#         for author in space.categories:
#             archivos = space.virtual_classes_holder_train[author].cat_file_list
#             #print author
#             
#             total_terms_in_class = 0
#             for arch in archivos:
#                 #print arch
# 
#                 tokens = space.virtual_classes_holder_train[author].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #space.virtual_classes_holder_train[author].dic_file_fd[arch]
                tamDoc = len(tokens)
                total_terms_in_class += tamDoc
                
                ################################################################
                # SUPER SPEED 
                
                for pal in docActualFd.keys_sorted():
                    
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
        self._ordered_new_labels_set =set_of_new_labels
        self._instance_subcategories = total_new_labels
        print "WWWWWWWWWWWWWWWW"
        #print self._instance_subcategories
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
            
        new_fdist = FreqDistExt()
        for term in new_vocabulary:
            new_fdist[term] = space._fdist[term]
        
        if debug:
            f_debug.write("after_fdist: " + str(new_fdist) + "\n")
            f_debug.write("final_vocab: " + str(new_fdist.keys_sorted()) + "\n")
            f_debug.close()
         
        # ----------------------------------------------------------------------
        
        space.update_and_save_fdist_and_vocabulary(new_fdist)
        
        return mat       

    def build_matrix(self):
        self.build_matrix_concepts_terms(self.space)
        self._matrix_concepts_terms0 = self.normalize_matrix_terms_concepts(self._matrix_concepts_terms0)
        
        if ('select_attributes' in self.space.kwargs_space):            
            self._matrix_concepts_terms0 = \
            self.__filter_mat_by_entropy_and_update_space_properties(self._matrix_concepts_terms0, 
                                                                     self.space)
                
        self.build_matrix_documents_concepts(self.space,
                                             self.space.virtual_classes_holder_train,
                                             self.space.corpus_file_list_train,
                                             self._matrix_concepts_terms0)
        
        #instances_matrix = self.get_matrix()
        #class_list = self.get_instance_categories()
        #namefiles_list = self.get_instance_namefiles()
        
        #self.perform_cluster_by_class(instances_matrix, class_list, namefiles_list)

        self.build_matrix_concepts_terms2(self.space)
        self._matrix_concepts_terms = self.normalize_matrix_terms_concepts(self._matrix_concepts_terms)
        
        self.build_matrix_documents_concepts(self.space,
                                             self.space.virtual_classes_holder_train,
                                             self.space.corpus_file_list_train,
                                             self._matrix_concepts_terms,
                                             SOA2=True)

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
        
    # ----------------------------------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._matrix_concepts_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_concepts_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass       
        
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            numpy.save(cache_file + "_mat_terms_concepts.npy", 
                       self.get_matrix_terms_concepts())
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())

            numpy.save(cache_file + "_ordered_new_labels_set.npy", 
                       self.get_ordered_new_labels_set())
            
            numpy.save(cache_file + "_instance_subcategories.npy", 
                       self.get_instance_subcategories())
            
            numpy.save(cache_file + "_dimensions_soa2.npy", 
                       self.get_dimensions_soa2())
        else:
            print "ERROR CSA: There is not a train matrix terms concepts built"
            
    def load_train_data(self, space):
        
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        csa2_train_matrix_holder = CSA2TrainMatrixHolder(space)        
        csa2_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))        
        csa2_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        csa2_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        csa2_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))  
        csa2_train_matrix_holder.set_ordered_new_labels_set(numpy.load(cache_file + "_ordered_new_labels_set.npy"))        
        csa2_train_matrix_holder.set_instance_subcategories(numpy.load(cache_file + "_instance_subcategories.npy"))        
        csa2_train_matrix_holder.set_dimensions_soa2(numpy.load(cache_file + "_dimensions_soa2.npy"))        
        
        return csa2_train_matrix_holder   


class CSA2TestMatrixHolder(CSA2MatrixHolder):

    def __init__(self, space, train_matrix_holder=None, matrix_concepts_terms=None):
        super(CSA2TestMatrixHolder, self).__init__(space)
        self._matrix_concepts_terms = numpy.transpose(train_matrix_holder.get_matrix_terms())
        #self._matrix_concepts_terms = matrix_concepts_terms
        #self.build_matrix()

    def normalize_matrix(self, normalizer, matrix):
        pass

    def build_matrix(self):

        self.build_matrix_documents_concepts(self.space,
                                             self.space.virtual_classes_holder_test,
                                             self.space.corpus_file_list_test,
                                             self._matrix_concepts_terms,
                                             SOA2=True)

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
        
    # ------------------------------------------------------------------------         
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._matrix_concepts_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_concepts_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass


class BOWMatrixHolder(MatrixHolder):

    def __init__(self, space):
        super(BOWMatrixHolder, self).__init__()
        self.space = space
        #self.build_matrix()

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
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] ##/ float(tamDoc)
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
        
    # ----------------------------------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass        
        
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            numpy.save(cache_file + "_mat_docs_terms.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
        else:
            print "ERROR BOW: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        bow_train_matrix_holder = BOWTrainMatrixHolder(space)        
        bow_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_terms.npy"))
        bow_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        bow_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return bow_train_matrix_holder 


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
        
    # ----------------------------------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass      
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass


class LSAMatrixHolder(MatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="???"):
        super(LSAMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None
        self.id2word = id2word
        self.tfidf = tfidf
        self.lsa = lsa
        self.corpus_tfidf = None
        self.corpus_lsa = None    
        self._id_dataset = dataset_label
        
    def get_tfidf(self):
        return self.tfidf
    
    def get_id2word(self):
        return self.id2word    
    
    def get_lsa(self):
        return self.lsa
    
    def set_tfidf(self, tfidf):
        self.tfidf = tfidf
    
    def set_id2word(self, id2word):
        self.id2word = id2word    
    
    def set_lsa(self, lsa):
        self.lsa = lsa
    
        
    def build_bowcorpus_id2word(self,
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
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/lsa")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/lsa/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/lsa/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)
            
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

class LSATrainMatrixHolder(LSAMatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="train"):
        super(LSATrainMatrixHolder, self).__init__(space, id2word, tfidf, lsa, dataset_label)
        self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_train, self.space.corpus_file_list_train)
        #self._id_dataset="train"
        if ('concepts' in self.space.kwargs_space):        
            self.dimensions = self.space.kwargs_space['concepts']
        else:
            self.dimensions = 300
        
    def build_matrix(self):
        
        dimensions = self.dimensions
        
#         if ('concepts' in self.space.kwargs_space):        
#             self.dimensions = self.space.kwargs_space['concepts']
#         else:
#             self.dimensions = 300
            
        #print self.corpus_bow
         
        self.tfidf = models.TfidfModel(self.corpus_bow) # step 1 -- initialize a model
        self.corpus_tfidf = self.tfidf[self.corpus_bow]
        self.lsa = models.LsiModel(self.corpus_tfidf, id2word=self.id2word, num_topics=dimensions) # run distributed LSA on documents
        self.corpus_lsa = self.lsa[self.corpus_tfidf]   
        
        #From sparse to dense
        matrix_documents_concepts = numpy.zeros((len(self.corpus_lsa), dimensions),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_lsa in self.corpus_lsa:            
            for (index, contribution) in doc_lsa:                
                matrix_documents_concepts[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_concepts
        #End of from sparse to dense                         

    def normalize_matrix(self):
        pass   
        
# 
#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_train,
#                                        self.space.corpus_file_list_train)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_train,
#                                                        self.space.corpus_file_list_train,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_train,
#                        self.space.corpus_file_list_train,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix        
        termcorpus = Dense2Corpus(self.get_lsa().projection.u.T)     
        #print list(termcorpus)
        
        #From sparse to dense
        matrix_terms_concepts = numpy.zeros((len(termcorpus), self.dimensions),
                                        dtype=numpy.float64)       
        cont_term=0
        for term_lsa in termcorpus:            
            for (index, contribution) in term_lsa:                
                matrix_terms_concepts[cont_term, index] = contribution            
            cont_term += 1
            
        #End of from sparse to dense  
            
        return matrix_terms_concepts
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
            
            id2word = self.get_id2word()            
            with open(space.space_path + "/lsa/" + space.id_space + "_id2word.txt", 'w') as outfile:
                json.dump(id2word, outfile)  
                          
            tfidf = self.get_tfidf()
            tfidf.save(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")
            
            lsa = self.get_lsa()
            lsa.save(space.space_path + "/lsa/" + space.id_space + "_model.lsi") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
            
            #self.set_matrix_terms(self)   # is this really necessary???
            
        else:
            print "ERROR LSA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        lsa_train_matrix_holder = LSATrainMatrixHolder(space)
        
        with open(space.space_path + "/lsa/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")      
        lsa = models.LsiModel.load(space.space_path + "/lsa/" + space.id_space + "_model.lsi")    
        
        lsa_train_matrix_holder.set_id2word(id2word)
        lsa_train_matrix_holder.set_tfidf(tfidf)
        lsa_train_matrix_holder.set_lsa(lsa)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        lsa_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        lsa_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        lsa_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return lsa_train_matrix_holder


class LSATestMatrixHolder(LSAMatrixHolder):

    def __init__(self, 
                 space, 
                 id2word=None, 
                 tfidf=None, 
                 lsa=None, 
                 train_matrix_holder=None, 
                 dataset_label="test"):
        
        train_matrix_holder = self.load_train_data(space)
        
        super(LSATestMatrixHolder, self).__init__(space, 
                                                  train_matrix_holder.id2word, 
                                                  train_matrix_holder.tfidf, 
                                                  train_matrix_holder.lsa, 
                                                  dataset_label)
        
        self.build_bowcorpus_id2word(self.space, 
                                     self.space.virtual_classes_holder_test, 
                                     self.space.corpus_file_list_test)
        
        if ('concepts' in self.space.kwargs_space):        
            self.dimensions = self.space.kwargs_space['concepts']
        else:
            self.dimensions = 300
        #self._id_dataset="test"
        
        
    def build_matrix(self):
        
        dimensions = self.dimensions
        
        self.corpus_tfidf = self.tfidf[self.corpus_bow]
        self.corpus_lsa = self.lsa[self.corpus_tfidf]        
        
        #From sparse to dense
        matrix_documents_concepts = numpy.zeros((len(self.corpus_lsa), dimensions),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_lsa in self.corpus_lsa:            
            for (index, contribution) in doc_lsa:                
                matrix_documents_concepts[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_concepts
        #End of from sparse to dense    

    def normalize_matrix(self):
        pass

#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_test,
#                                        self.space.corpus_file_list_test)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_test,
#                                                        self.space.corpus_file_list_test,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_test,
#                        self.space.corpus_file_list_test,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix        
        termcorpus = Dense2Corpus(self.get_lsa().projection.u.T)     
        #print list(termcorpus)
        
        #From sparse to dense
        matrix_terms_concepts = numpy.zeros((len(termcorpus), self.dimensions),
                                        dtype=numpy.float64)       
        cont_term=0
        for term_lsa in termcorpus:            
            for (index, contribution) in term_lsa:                
                matrix_terms_concepts[cont_term, index] = contribution            
            cont_term += 1
            
        #End of from sparse to dense  
        
        return matrix_terms_concepts
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        lsa_train_matrix_holder = LSATrainMatrixHolder(space)
        
        with open(space.space_path + "/lsa/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")      
        lsa = models.LsiModel.load(space.space_path + "/lsa/" + space.id_space + "_model.lsi")    
        
        lsa_train_matrix_holder.set_id2word(id2word)
        lsa_train_matrix_holder.set_tfidf(tfidf)
        lsa_train_matrix_holder.set_lsa(lsa)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        lsa_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        lsa_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        lsa_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return lsa_train_matrix_holder
        
        

class TFIDFMatrixHolder(MatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="???"):
        super(TFIDFMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None
        self.id2word = id2word
        self.tfidf = tfidf
        self.lsa = lsa
        self.corpus_tfidf = None
        self.corpus_lsa = None    
        self._id_dataset = dataset_label
        
    def get_tfidf(self):
        return self.tfidf
    
    def get_id2word(self):
        return self.id2word    
    
    def get_lsa(self):
        return self.lsa
    
    def set_tfidf(self, tfidf):
        self.tfidf = tfidf
    
    def set_id2word(self, id2word):
        self.id2word = id2word    
    
    def set_lsa(self, lsa):
        self.lsa = lsa
    
        
    def build_bowcorpus_id2word(self,
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
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/tfidf")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/tfidf/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/tfidf/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)
            
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

class TFIDFTrainMatrixHolder(TFIDFMatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="train"):
        super(TFIDFTrainMatrixHolder, self).__init__(space, id2word, tfidf, lsa, dataset_label)
        self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_train, self.space.corpus_file_list_train)
        #self._id_dataset="train"
        if ('concepts' in self.space.kwargs_space):        
            self.dimensions = self.space.kwargs_space['concepts']
        else:
            self.dimensions = 300
        
    def build_matrix(self):
        
        dimensions = self.dimensions
        
#         if ('concepts' in self.space.kwargs_space):        
#             self.dimensions = self.space.kwargs_space['concepts']
#         else:
#             self.dimensions = 300
            
        #print self.corpus_bow
         
        self.tfidf = models.TfidfModel(self.corpus_bow) # step 1 -- initialize a model
        self.corpus_tfidf = self.tfidf[self.corpus_bow]
        # self.lsa = models.LsiModel(self.corpus_tfidf, id2word=self.id2word, num_topics=dimensions) # run distributed LSA on documents
        # self.corpus_lsa = self.lsa[self.corpus_tfidf]   
        
        #print len(self.corpus_tfidf)
        #print  self.space._vocabulary
        #From sparse to dense
        matrix_documents_tfidf = numpy.zeros((len(self.corpus_tfidf), len(self.space._vocabulary)),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_tfidf in self.corpus_tfidf:            
            for (index, contribution) in doc_tfidf:                
                matrix_documents_tfidf[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_tfidf
        #End of from sparse to dense                         

    def normalize_matrix(self):
        pass   
        
# 
#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_train,
#                                        self.space.corpus_file_list_train)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_train,
#                                                        self.space.corpus_file_list_train,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_train,
#                        self.space.corpus_file_list_train,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix        
        termcorpus = Dense2Corpus(self.get_lsa().projection.u.T)     
        #print list(termcorpus)
        
        #From sparse to dense
        matrix_terms_concepts = numpy.zeros((len(termcorpus), self.dimensions),
                                        dtype=numpy.float64)       
        cont_term=0
        for term_lsa in termcorpus:            
            for (index, contribution) in term_lsa:                
                matrix_terms_concepts[cont_term, index] = contribution            
            cont_term += 1
            
        #End of from sparse to dense  
            
        return matrix_terms_concepts
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
            
            id2word = self.get_id2word()            
            with open(space.space_path + "/tfidf/" + space.id_space + "_id2word.txt", 'w') as outfile:
                json.dump(id2word, outfile)  
                          
            tfidf = self.get_tfidf()
            tfidf.save(space.space_path + "/tfidf/" + space.id_space + "_model.tfidf")
            
            #lsa = self.get_lsa()
            #lsa.save(space.space_path + "/lsa/" + space.id_space + "_model.lsi") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
            
            #self.set_matrix_terms(self)   # is this really necessary???
            
        else:
            print "ERROR LSA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        tfidf_train_matrix_holder = TFIDFTrainMatrixHolder(space)
        
        with open(space.space_path + "/tfidf/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/tfidf/" + space.id_space + "_model.tfidf")      
        # lsa = models.LsiModel.load(space.space_path + "/tfidf/" + space.id_space + "_model.lsi")    
        
        tfidf_train_matrix_holder.set_id2word(id2word)
        tfidf_train_matrix_holder.set_tfidf(tfidf)
        # tfidf_train_matrix_holder.set_lsa(lsa)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        tfidf_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        tfidf_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        tfidf_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #tfidf_train_matrix_holder.set_matrix_terms(tfidf_train_matrix_holder)   # this is necessary
        
        return tfidf_train_matrix_holder


class TFIDFTestMatrixHolder(TFIDFMatrixHolder):

    def __init__(self, 
                 space, 
                 id2word=None, 
                 tfidf=None, 
                 lsa=None, 
                 train_matrix_holder=None, 
                 dataset_label="test"):
        
        train_matrix_holder = self.load_train_data(space)
        
        super(TFIDFTestMatrixHolder, self).__init__(space, 
                                                  train_matrix_holder.id2word, 
                                                  train_matrix_holder.tfidf, 
                                                  train_matrix_holder.lsa, 
                                                  dataset_label)
        
        self.build_bowcorpus_id2word(self.space, 
                                     self.space.virtual_classes_holder_test, 
                                     self.space.corpus_file_list_test)
        
        if ('concepts' in self.space.kwargs_space):        
            self.dimensions = self.space.kwargs_space['concepts']
        else:
            self.dimensions = 300
        #self._id_dataset="test"
        
        
    def build_matrix(self):
        
        dimensions = self.dimensions
        
        self.corpus_tfidf = self.tfidf[self.corpus_bow]
        # self.corpus_lsa = self.lsa[self.corpus_tfidf]        
        
        #From sparse to dense
        matrix_documents_tfidf = numpy.zeros((len(self.corpus_tfidf), len(self.space._vocabulary)),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_tfidf in self.corpus_tfidf:            
            for (index, contribution) in doc_tfidf:                
                matrix_documents_tfidf[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_tfidf
        #End of from sparse to dense    

    def normalize_matrix(self):
        pass

#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_test,
#                                        self.space.corpus_file_list_test)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_test,
#                                                        self.space.corpus_file_list_test,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_test,
#                        self.space.corpus_file_list_test,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix        
        termcorpus = Dense2Corpus(self.get_lsa().projection.u.T)     
        #print list(termcorpus)
        
        #From sparse to dense
        matrix_terms_concepts = numpy.zeros((len(termcorpus), self.dimensions),
                                        dtype=numpy.float64)       
        cont_term=0
        for term_lsa in termcorpus:            
            for (index, contribution) in term_lsa:                
                matrix_terms_concepts[cont_term, index] = contribution            
            cont_term += 1
            
        #End of from sparse to dense  
        
        return matrix_terms_concepts
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        tfidf_train_matrix_holder = TFIDFTrainMatrixHolder(space)
        
        with open(space.space_path + "/tfidf/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/tfidf/" + space.id_space + "_model.tfidf")      
        # lsa = models.LsiModel.load(space.space_path + "/tfidf/" + space.id_space + "_model.lsi")    
        
        tfidf_train_matrix_holder.set_id2word(id2word)
        tfidf_train_matrix_holder.set_tfidf(tfidf)
        # tfidf_train_matrix_holder.set_lsa(lsa)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        tfidf_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        tfidf_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        tfidf_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #tfidf_train_matrix_holder.set_matrix_terms(tfidf_train_matrix_holder)   # this is necessary
        
        return tfidf_train_matrix_holder
        
########### DOR ############

class DORMatrixHolder(MatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="???"):
        super(DORMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None
        self.id2word = id2word
        self.tfidf = tfidf
        self.lsa = lsa
        self.corpus_tfidf = None
        self.corpus_lsa = None    
        self._id_dataset = dataset_label
        
    def get_tfidf(self):
        return self.tfidf
    
    def get_id2word(self):
        return self.id2word    
    
    def get_lsa(self):
        return self.lsa
    
    def set_tfidf(self, tfidf):
        self.tfidf = tfidf
    
    def set_id2word(self, id2word):
        self.id2word = id2word    
    
    def set_lsa(self, lsa):
        self.lsa = lsa
        
    def set_mat_docs_terms(self, mat):
        self._mat_docs_terms = mat
        
    def get_mat_docs_terms(self):
        return self._mat_docs_terms
    
        
    def build_bowcorpus_id2word(self,
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
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                tam_V=len(unorder_dict_index)
                tam_v=len(docActualFd)
                print "tam_V: " + str(tam_V)
                print "tam_v: " + str(tam_v)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        matrix_docs_terms[i, unorder_dict_index[pal]] = (1 + math.log10(freq)) * math.log10(tam_V/tam_v)
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]
        
        print "------"
        print matrix_docs_terms
        print "------"
        
        #matrix_docs_terms=matrix_docs_terms ** 2 # matrix_docs_terms
        norma=numpy.sqrt( ( matrix_docs_terms ** 2 ).sum(axis=0) ) # sum all columns
        norma+=0.00000001
        matrix_docs_terms=matrix_docs_terms/norma
        
        print "+++++"
        print matrix_docs_terms
        print "+++++"
        
        self._mat_docs_terms = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)
        
        
    def build_matrix_dor(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_docs_terms):

        t1 = time.time()
        print "Starting BOW representation..."
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_docs = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_docs = numpy.zeros((len(corpus_file_list), len(mat_docs_terms)),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] / float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        
                        matrix_docs_docs[i, :] += mat_docs_terms[:, unorder_dict_index[pal]] * freq #/tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
            
        #Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_docs
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of DOR representation. Time: ", str(t2-t1)
        
        
            
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

class DORTrainMatrixHolder(DORMatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="train"):
        super(DORTrainMatrixHolder, self).__init__(space, id2word, tfidf, lsa, dataset_label)
        #self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_train, self.space.corpus_file_list_train)
        #self._id_dataset="train"
        
    
    def build_matrix(self):
        self.build_matrix_dor(self.space,
                                             self.space.virtual_classes_holder_train,
                                             self.space.corpus_file_list_train,
                                             self._mat_docs_terms)
        
        
                                

    def normalize_matrix(self):
        pass   
        
# 
#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_train,
#                                        self.space.corpus_file_list_train)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_train,
#                                                        self.space.corpus_file_list_train,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_train,
#                        self.space.corpus_file_list_train,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------
        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._mat_docs_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._mat_docs_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
            
            id2word = self.get_id2word()            
            with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'w') as outfile:
                json.dump(id2word, outfile)  
                          
            #tfidf = self.__dor_train_matrix_holder.get_tfidf()
            #tfidf.save(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")
            
            #dor = self.__dor_train_matrix_holder.get_dor()
            #dor.save(space.space_path + "/dor/" + space.id_space + "_model.dor") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_docs.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_mat_docs_terms.npy", 
                       self.get_mat_docs_terms())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
        else:
            print "ERROR LSA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        dor_train_matrix_holder = DORTrainMatrixHolder(space)
        
        with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        #tfidf = models.TfidfModel.load(space.space_path + "/dor/" + space.id_space + "_model.tfidf")      
        #dor = models.LsiModel.load(space.space_path + "/dor/" + space.id_space + "_model.dor")    
        
        dor_train_matrix_holder.set_id2word(id2word)
        #self.__dor_train_matrix_holder.set_tfidf(tfidf)
        #self.__dor_train_matrix_holder.set_dor(dor)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        dor_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_docs.npy"))
        dor_train_matrix_holder.set_mat_docs_terms(numpy.load(cache_file + "_mat_docs_terms.npy"))
        dor_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        dor_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return dor_train_matrix_holder  
        



class DORTestMatrixHolder(DORMatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, train_matrix_holder=None, dataset_label="test"):
        super(DORTestMatrixHolder, self).__init__(space, id2word, tfidf, lsa, dataset_label)
        self.set_matrix_terms(train_matrix_holder.get_matrix_terms())
        #self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_test, self.space.corpus_file_list_test)
        #self._id_dataset="test"
        
        
    def build_matrix(self):
        self.build_matrix_dor(self.space,
                                             self.space.virtual_classes_holder_test,
                                             self.space.corpus_file_list_test,
                                             self._mat_docs_terms) 

    def normalize_matrix(self):
        pass

#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_test,
#                                        self.space.corpus_file_list_test)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_test,
#                                                        self.space.corpus_file_list_test,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_test,
#                        self.space.corpus_file_list_test,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------------------------    
        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._mat_docs_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._mat_docs_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    
########### DOR ############   

########### TCOR ############

class TCORMatrixHolder(MatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="???"):
        super(TCORMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None
        self.id2word = id2word
        self.tfidf = tfidf
        self.lsa = lsa
        self.corpus_tfidf = None
        self.corpus_lsa = None    
        self._id_dataset = dataset_label
        
    def get_tfidf(self):
        return self.tfidf
    
    def get_id2word(self):
        return self.id2word    
    
    def get_lsa(self):
        return self.lsa
    
    def set_tfidf(self, tfidf):
        self.tfidf = tfidf
    
    def set_id2word(self, id2word):
        self.id2word = id2word    
    
    def set_lsa(self, lsa):
        self.lsa = lsa
        
    def set_mat_docs_terms(self, mat):
        self._mat_docs_terms = mat
        
    def get_mat_terms_terms(self):
        return self._mat_terms_terms
    
        
    def build_bowcorpus_id2word(self,
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
            matrix_terms_terms = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_terms_terms = numpy.zeros((len_vocab, len_vocab),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                tam_V=len(unorder_dict_index)
                tam_v=len(docActualFd)
                print "tam_V: " + str(tam_V)
                print "tam_v: " + str(tam_v)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        # matrix_terms_terms[i, unorder_dict_index[pal]] = (1 + math.log10(freq)) * math.log10(tam_V/tam_v)
                    
                    if freq > 0.0:
                        rows_file.write(str(i) + "\n")
                        columns_file.write(str(unorder_dict_index[pal]) + "\n")
                        vals_file.write(str(freq) + "\n")
                        
                # populate the coocurrences with other terms
                for w1 in set(docActualFd.keys_sorted()):
                    
                    for w2 in set(docActualFd.keys_sorted()):
                        
                        matrix_terms_terms[unorder_dict_index[w2], unorder_dict_index[w1]] += 1.0
                        
                # populate the coocurrences with itself
                #for w1 in set(docActualFd.keys_sorted()):
                    
                #    if docActualFd[w1] >= 2:
                #        matrix_terms_terms[unorder_dict_index[w1], unorder_dict_index[w1]] += 1.0
                    
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
                
                corpus_bow += [bow]
                
        numpy.set_printoptions(precision=3)
        print matrix_terms_terms
        TK = []
        for tw in range(tam_V):
            print "term: ",tw
            print tam_V
            print float(numpy.size(numpy.where(matrix_terms_terms[tw,:] > 0), axis=1))
            TK += [tam_V/float(numpy.size(numpy.where(matrix_terms_terms[tw,:] > 0), axis=1))]
            
        TKa = numpy.array(TK) 
        
        print "TKa"
        print TKa
        
        print "numpy.log10(TKa)"
        print numpy.log10(TKa)
                
        matrix_terms_terms[numpy.where(matrix_terms_terms > 0)] = (1 + numpy.log10(matrix_terms_terms[numpy.where(matrix_terms_terms > 0)]))  
        
        print "matrix_terms_terms[numpy.where(matrix_terms_terms > 0)] = (1 + numpy.log10(matrix_terms_terms[numpy.where(matrix_terms_terms > 0)]))"
        print matrix_terms_terms 
        
        matrix_terms_terms = numpy.transpose(numpy.transpose(matrix_terms_terms) * numpy.log10(TKa))
        
        print "matrix_terms_terms = numpy.transpose(numpy.transpose(matrix_terms_terms) * numpy.log10(TKa))"
        print matrix_terms_terms
        
        Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]
        
        print "------"
        # print matrix_terms_terms
        print "------"
        
        #matrix_docs_terms=matrix_docs_terms ** 2 # matrix_docs_terms
        norma=numpy.sqrt( ( matrix_terms_terms ** 2 ).sum(axis=0) ) # sum all columns
        print "norma=numpy.sqrt( ( matrix_terms_terms ** 2 ).sum(axis=0) ) # sum all columns"
        print norma
        norma[numpy.where(norma == 0)] += 0.00000001
        print "norma[numpy.where(norma == 0)] += 0.00000001"
        print norma
        matrix_terms_terms=matrix_terms_terms/norma
        
        print "+++++"
        print matrix_terms_terms
        print "+++++"
        
        self._mat_terms_terms = matrix_terms_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)
        
        
    def build_matrix_dor(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list,
                          mat_docs_terms):

        t1 = time.time()
        print "Starting BOW representation..."
        
        len_vocab = len(space._vocabulary)

        Util.create_a_dir(space.space_path + "/sparse")
        rows_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "rows_sparse.txt", "w")
        columns_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "columns_sparse.txt", "w")
        vals_file = open(space.space_path + "/sparse/" + space.id_space + "_" + "vals_sparce.txt", "w")
        
        dense_flag = True
        
        if ('sparse' in space.kwargs_space) and space.kwargs_space['sparse']:            
            matrix_docs_docs = numpy.zeros((1, 1),
                                        dtype=numpy.float64)
            dense_flag = False
        else:
            matrix_docs_docs = numpy.zeros((len(corpus_file_list), len(mat_docs_terms)),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] / float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        
                        
                        #print matrix_docs_docs
                        
                        #print "##########################" + str(len(mat_docs_terms))
                        
                        #print mat_docs_terms
                        
                        #print unorder_dict_index[pal]
                        
                        #print "##########################MAT_DOCS_TERMS: " + str(len(mat_docs_terms[:, unorder_dict_index[pal]]))
                        #print "##########################MAT_DOCS_TERMS_T: " + str(len((mat_docs_terms[:, unorder_dict_index[pal]].transpose()), axis=1))
                        
                        #print "##########################MAT_DOCS_DOCS: " + str(len(matrix_docs_docs[i, :]))
                        
                        matrix_docs_docs[i, :] += mat_docs_terms[:, unorder_dict_index[pal]] * freq #/ tamDoc
                        
                        
                    
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
                
                corpus_bow += [bow]
            
        #Util.create_a_dir(space.space_path + "/dor")
        
        #print corpus_bow
            
        #corpora.MmCorpus.serialize(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        #self.corpus_bow = corpora.MmCorpus(space.space_path + "/dor/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_docs
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of DOR representation. Time: ", str(t2-t1)
        
        
            
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

class TCORTrainMatrixHolder(TCORMatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, dataset_label="train"):
        super(TCORTrainMatrixHolder, self).__init__(space, id2word, tfidf, lsa, dataset_label)
        #self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_train, self.space.corpus_file_list_train)
        #self._id_dataset="train"
        
    
    def build_matrix(self):
        self.build_matrix_dor(self.space,
                                             self.space.virtual_classes_holder_train,
                                             self.space.corpus_file_list_train,
                                             self._mat_terms_terms)
        
        
                                

    def normalize_matrix(self):
        pass   
        
# 
#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_train,
#                                        self.space.corpus_file_list_train)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_train,
#                                                        self.space.corpus_file_list_train,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_train,
#                        self.space.corpus_file_list_train,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------
        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._mat_terms_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._mat_terms_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
            
            id2word = self.get_id2word()            
            with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'w') as outfile:
                json.dump(id2word, outfile)  
                          
            #tfidf = self.__dor_train_matrix_holder.get_tfidf()
            #tfidf.save(space.space_path + "/lsa/" + space.id_space + "_model.tfidf")
            
            #dor = self.__dor_train_matrix_holder.get_dor()
            #dor.save(space.space_path + "/dor/" + space.id_space + "_model.dor") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_docs.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_mat_terms_terms.npy", 
                       self.get_mat_terms_terms())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
        else:
            print "ERROR LSA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        dor_train_matrix_holder = DORTrainMatrixHolder(space)
        
        with open(space.space_path + "/dor/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        #tfidf = models.TfidfModel.load(space.space_path + "/dor/" + space.id_space + "_model.tfidf")      
        #dor = models.LsiModel.load(space.space_path + "/dor/" + space.id_space + "_model.dor")    
        
        dor_train_matrix_holder.set_id2word(id2word)
        #self.__dor_train_matrix_holder.set_tfidf(tfidf)
        #self.__dor_train_matrix_holder.set_dor(dor)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        dor_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_docs.npy"))
        dor_train_matrix_holder.set_mat_docs_terms(numpy.load(cache_file + "_mat_terms_terms.npy"))
        dor_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        dor_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return dor_train_matrix_holder  
        



class TCORTestMatrixHolder(TCORMatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lsa=None, train_matrix_holder=None, dataset_label="test"):
        super(TCORTestMatrixHolder, self).__init__(space, id2word, tfidf, lsa, dataset_label)
        self.set_matrix_terms(train_matrix_holder.get_matrix_terms())
        #self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_test, self.space.corpus_file_list_test)
        #self._id_dataset="test"
        
        
    def build_matrix(self):
        self.build_matrix_dor(self.space,
                                             self.space.virtual_classes_holder_test,
                                             self.space.corpus_file_list_test,
                                             self._mat_terms_terms) 

    def normalize_matrix(self):
        pass

#     def build_matrix(self):
#         matrix_docs_terms = self.build_matrix_doc_terminos(self.space,
#                                        self.space.virtual_classes_holder_test,
#                                        self.space.corpus_file_list_test)
# 
#         weighted_matrix_docs_terms = self.build_weight(self.space,
#                                                        self.space.virtual_classes_holder_test,
#                                                        self.space.corpus_file_list_test,
#                                                        matrix_docs_terms)
#         self.build_lsi(self.space,
#                        self.space.virtual_classes_holder_test,
#                        self.space.corpus_file_list_test,
#                        weighted_matrix_docs_terms)

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
        
    # ------------------------------------------------------------------------    
        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix
        return numpy.transpose(self._mat_docs_terms)
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._mat_docs_terms = numpy.transpose(value)
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        pass
    
########### TCOR ############        
      
class LDAMatrixHolder(MatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lda=None, dataset_label="???"):
        super(LDAMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None
        self.id2word = id2word
        self.tfidf = tfidf
        self.lda = lda
        self.corpus_tfidf = None
        self.corpus_lda = None    
        self._id_dataset = dataset_label
        
    def get_tfidf(self):
        return self.tfidf
    
    def get_id2word(self):
        return self.id2word    
    
    def get_lda(self):
        return self.lda
    
    def set_tfidf(self, tfidf):
        self.tfidf = tfidf
    
    def set_id2word(self, id2word):
        self.id2word = id2word    
    
    def set_lda(self, lda):
        self.lda = lda
    
        
    def build_bowcorpus_id2word(self,
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
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/lda")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/lda/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/lda/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lda = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LDA on documents
        #corpus_lda = lda[corpus_tfidf]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)
            
    def build_weight(self,
                     space,
                     virtual_classes_holder,
                     corpus_file_list):

        weighted_matrix_docs_terms = None

        #some code

        return weighted_matrix_docs_terms


class LDATrainMatrixHolder(LDAMatrixHolder):

    def __init__(self, space, id2word=None, tfidf=None, lda=None, dataset_label="train"):
        super(LDATrainMatrixHolder, self).__init__(space, id2word, tfidf, lda, dataset_label)
        self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_train, self.space.corpus_file_list_train)
        #self._id_dataset="train"
        
    def build_matrix(self):
        
        if ('concepts' in self.space.kwargs_space):        
            dimensions = self.space.kwargs_space['concepts']
        else:
            dimensions = 300
            
        #print self.corpus_bow
        
        # UNUSED ######################################## 
        self.tfidf = models.TfidfModel(self.corpus_bow) # step 1 -- initialize a model
        self.corpus_tfidf = self.tfidf[self.corpus_bow]
        #################################################
        
        self.lda = models.LdaModel(self.corpus_bow, id2word=self.id2word, num_topics=dimensions) # run distributed LDA on documents
        self.corpus_lda = self.lda[self.corpus_bow]   
        
        #From sparse to dense
        matrix_documents_concepts = numpy.zeros((len(self.corpus_lda), dimensions),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_lda in self.corpus_lda:            
            for (index, contribution) in doc_lda:                
                matrix_documents_concepts[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_concepts
        #End of from sparse to dense                             

    def normalize_matrix(self):
        pass   
        

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
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix   
        pass 
#         termcorpus = Dense2Corpus(self.get_lda().projection.u.T)     
#         #print list(termcorpus)
#         
#         #From sparse to dense
#         matrix_terms_concepts = numpy.zeros((len(termcorpus), self.dimensions),
#                                         dtype=numpy.float64)       
#         cont_term=0
#         for term_lda in termcorpus:            
#             for (index, contribution) in term_lda:                
#                 matrix_terms_concepts[cont_term, index] = contribution            
#             cont_term += 1
#             
#         #End of from sparse to dense  
#             
#         return matrix_terms_concepts
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lda_train_matrix_holder.get_matrix_terms_concepts())
            
            id2word = self.get_id2word()            
            with open(space.space_path + "/lda/" + space.id_space + "_id2word.txt", 'w') as outfile:
                json.dump(id2word, outfile)  
                          
            tfidf = self.get_tfidf()
            tfidf.save(space.space_path + "/lda/" + space.id_space + "_model.tfidf")
            
            lda = self.get_lda()
            lda.save(space.space_path + "/lda/" + space.id_space + "_model.lda") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
        else:
            print "ERROR LDA: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        lda_train_matrix_holder = LDATrainMatrixHolder(space)
        
        with open(space.space_path + "/lda/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)       
        tfidf = models.TfidfModel.load(space.space_path + "/lda/" + space.id_space + "_model.tfidf")      
        lda = models.LdaModel.load(space.space_path + "/lda/" + space.id_space + "_model.lda")    
        
        lda_train_matrix_holder.set_id2word(id2word)
        lda_train_matrix_holder.set_tfidf(tfidf)
        lda_train_matrix_holder.set_lda(lda)
          
        #self.__lda_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        lda_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        lda_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        lda_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return lda_train_matrix_holder 


class LDATestMatrixHolder(LDAMatrixHolder):

    def __init__(self, 
                 space, 
                 id2word=None, 
                 tfidf=None, 
                 lda=None, 
                 train_matrix_holder=None, 
                 dataset_label="test"):
        
        train_matrix_holder = self.load_train_data(space)
        
        super(LDATestMatrixHolder, self).__init__(space, 
                                                  train_matrix_holder.id2word, 
                                                  train_matrix_holder.tfidf, 
                                                  train_matrix_holder.lda, 
                                                  dataset_label)
        
        self.build_bowcorpus_id2word(self.space, self.space.virtual_classes_holder_test, self.space.corpus_file_list_test)
        #self._id_dataset="test"
        
    def build_matrix(self):
        
        if ('concepts' in self.space.kwargs_space):        
            dimensions = self.space.kwargs_space['concepts']
        else:
            dimensions = 300
        
        # UNUSED ########################################
        self.corpus_tfidf = self.tfidf[self.corpus_bow]
        #################################################
        
        self.corpus_lda = self.lda[self.corpus_bow]        
        
        #From sparse to dense
        matrix_documents_concepts = numpy.zeros((len(self.corpus_lda), dimensions),
                                        dtype=numpy.float64)       
        cont_doc=0
        for doc_lda in self.corpus_lda:            
            for (index, contribution) in doc_lda:                
                matrix_documents_concepts[cont_doc, index] = contribution            
            cont_doc += 1
            
        self._matrix = matrix_documents_concepts
        #End of from sparse to dense    

    def normalize_matrix(self):
        pass

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
        
    def save_train_data(self, space):
        pass
    
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix      
        pass  
#         termcorpus = Dense2Corpus(self.get_lda().projection.u.T)     
#         #print list(termcorpus)
#         
#         #From sparse to dense
#         matrix_terms_concepts = numpy.zeros((len(termcorpus), self.dimensions),
#                                         dtype=numpy.float64)       
#         cont_term=0
#         for term_lda in termcorpus:            
#             for (index, contribution) in term_lda:                
#                 matrix_terms_concepts[cont_term, index] = contribution            
#             cont_term += 1
#             
#         #End of from sparse to dense  
#         
#         return matrix_terms_concepts
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        lda_train_matrix_holder = LDATrainMatrixHolder(space)
        
        with open(space.space_path + "/lda/" + space.id_space + "_id2word.txt", 'r') as infile:
                id2word = json.load(infile)
        tfidf = models.TfidfModel.load(space.space_path + "/lda/" + space.id_space + "_model.tfidf")      
        lda = models.LdaModel.load(space.space_path + "/lda/" + space.id_space + "_model.lda")    
        
        lda_train_matrix_holder.set_id2word(id2word)
        lda_train_matrix_holder.set_tfidf(tfidf)
        lda_train_matrix_holder.set_lda(lda)
          
        #self.__lda_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        lda_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        lda_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        lda_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))      
        
        return lda_train_matrix_holder 
    

class W2VVLADMatrixHolder(MatrixHolder):

    def __init__(self, space, w2v=None, dataset_label="???"):
        super(W2VVLADMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None    
        self._id_dataset = dataset_label
        self._train_sentences= None
        self._train_model = w2v
        self._matrix_terms_dimensions = None
        
        if ('concepts' in self.space.kwargs_space):        
            self.dimensions = self.space.kwargs_space['concepts']
        else:
            self.dimensions = 100
            
        if ('min_count' in self.space.kwargs_space):        
            self.min_count = self.space.kwargs_space['min_count']
        else:
            self.min_count = 1
            
        if ('workers' in self.space.kwargs_space):        
            self.workers = self.space.kwargs_space['workers']
        else:
            self.workers = 4
            
        if 'w2v_txt' in self.space.kwargs_space:
            self._w2v_txt = self.space.kwargs_space['w2v_txt']
        else:
            self._w2v_txt = "NOT_PROVIDED"
        
    def get_w2v(self):
        return self._train_model
    
    def set_w2v(self, value):
        self._train_model = value
        
    def build_naive_representation(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list):

        t1 = time.time()
        print "Starting W2VVLADMatrixHolder representation..."
        
        dimensions = self.dimensions
        
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
            matrix_docs_terms = numpy.zeros((len(corpus_file_list), dimensions * self.space.kwargs_space['vlad_dim']),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        mat_terms_dimensions = numpy.zeros((len_vocab, dimensions), dtype=numpy.float64)
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            if term in self._train_model:
                mat_terms_dimensions[u, :] = self._train_model[term]
        self._matrix_terms_dimensions = mat_terms_dimensions
        ###############################################################
      
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                # begining the block to build the arrangament of w2v vectors to be encoded by other script in matlab
                Util.create_a_dir(space.space_path + "/tmp_matlab")
                pathDataToBeEncoded = space.space_path + "/tmp_matlab/" + space.id_space + "_" + "tmpFileDataToBeEncoded.csv"
                tmpFileDataToBeEncoded = open(pathDataToBeEncoded, "w")
                
                for token in tokens:
                    if token in tokens:
                        w2v_vec =  self._train_model[token]
                        tmpFileDataToBeEncoded.write(",".join(map(str, w2v_vec))+"\n")
                tmpFileDataToBeEncoded.close()
                
                path_for_enc = space.space_path + "/tmp_matlab/" + space.id_space + "_" + "enc.csv" 
                cmd = ("matlab -nodisplay -nodesktop -nosplash -r \"try, clear all; clc; close all; dataToBeEncoded='" + 
                pathDataToBeEncoded + "' ; path_env_file='" + self.space.kwargs_space['env_file'] + "' ;cwd='$PWD" 
                "' ; numClusters=" + str(self.space.kwargs_space['vlad_dim']) + "; path_for_enc='" + path_for_enc + 
                "' ; run('/media/aplm/Extension/aplm/Dropbox/MATLABProjects/FisherVectors/step2_vlad.m'); end;quit\"")
                print cmd
                print i
                subprocess.call(cmd,shell=True)        
                enc = numpy.genfromtxt(path_for_enc, delimiter=',')
                if tamDoc > 0:
                    matrix_docs_terms[i, :] = enc
                
                # ending the block to build the arrangament of w2v vectors to be encoded by other script in matlab
                
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        if pal in self._train_model:
                            pass
                            #matrix_docs_terms[i, :] += self._train_model[pal] * freq
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/w2v")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of W2VMatrixHolder representation. Time: ", str(t2-t1)
        
    def build_collect_train_sentences(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list):


        t1 = time.time()
        print "Starting build_collect_train_sentences ..."
        
        self._train_sentences = []
        
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
            #matrix_docs_terms = numpy.zeros((len(corpus_file_list), len_vocab),
            #                            dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                
                self._train_sentences += [tokens]
                
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/w2v")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        #self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of build_collect_train_sentences Time: ", str(t2-t1)
            

class W2VVLADTrainMatrixHolder(W2VVLADMatrixHolder):

    def __init__(self, space, dataset_label="train", train=True):
        super(W2VVLADTrainMatrixHolder, self).__init__(space, w2v=None, dataset_label=dataset_label)
        
        if self._w2v_txt == "NOT_PROVIDED":
            self.build_collect_train_sentences(self.space, 
                                               self.space.virtual_classes_holder_train,
                                               self.space.corpus_file_list_train)
            
            
        if train == True:
            self.train_w2v(self._train_sentences)
            
    def train_w2v(self, sentences):
        
        # print self._train_sentences  
        
        if self._w2v_txt == "NOT_PROVIDED":      
            self._train_model = Word2Vec(self._train_sentences, 
                                         size=self.dimensions, 
                                         min_count=self.min_count, 
                                         workers=self.workers)
        else:
            print "Loading w2v file model ..."
            self._train_model = Word2Vec.load_word2vec_format(self._w2v_txt, binary=False)
            print "End of loading w2v file model ..."   
        
    def build_matrix(self):                
        self.build_naive_representation(self.space,
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
        
    def normalize_matrix(self, normalizer, matrix):
        pass
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix                     
        return self._matrix_terms_dimensions
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_terms_dimensions = value
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
                          
            w2v = self.get_w2v()
            w2v.save(space.space_path + "/w2v/" + space.id_space + "_w2v_model") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
            
            #self.set_matrix_terms(self)   # is this really necessary???
            
        else:
            print "ERROR W2V: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        w2v_train_matrix_holder = W2VTrainMatrixHolder(space, train=False) 
      
        w2v = Word2Vec()
        w2v = w2v.load(space.space_path + "/w2v/" + space.id_space + "_w2v_model")    
        
        w2v_train_matrix_holder.set_w2v(w2v)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        w2v_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        w2v_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        w2v_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return w2v_train_matrix_holder


class W2VVLADTestMatrixHolder(W2VVLADMatrixHolder):

    def __init__(self, 
                 space,
                 train_matrix_holder=None, 
                 dataset_label="test"):
        
        train_matrix_holder = self.load_train_data(space)
        
        super(W2VVLADTestMatrixHolder, self).__init__(space, 
                                                  w2v=train_matrix_holder.get_w2v(),
                                                  dataset_label=dataset_label)     
        
    def build_matrix(self):
        
        self.build_naive_representation(self.space,
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
        
    def normalize_matrix(self, normalizer, matrix):
        pass
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix                     
        return self._matrix_terms_dimensions
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_terms_dimensions = value
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        w2v_train_matrix_holder = W2VTrainMatrixHolder(space, train=False)
              
        w2v = Word2Vec()
        w2v = w2v.load(space.space_path + "/w2v/" + space.id_space + "_w2v_model")    
        
        w2v_train_matrix_holder.set_w2v(w2v)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        w2v_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        w2v_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        w2v_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return w2v_train_matrix_holder
    

class W2VMatrixHolder(MatrixHolder):

    def __init__(self, space, w2v=None, dataset_label="???"):
        super(W2VMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None    
        self._id_dataset = dataset_label
        self._train_sentences= None
        self._train_model = w2v
        self._matrix_terms_dimensions = None
        
        if ('concepts' in self.space.kwargs_space):        
            self.dimensions = self.space.kwargs_space['concepts']
        else:
            self.dimensions = 100
            
        if ('min_count' in self.space.kwargs_space):        
            self.min_count = self.space.kwargs_space['min_count']
        else:
            self.min_count = 1
            
        if ('workers' in self.space.kwargs_space):        
            self.workers = self.space.kwargs_space['workers']
        else:
            self.workers = 4
            
        if 'w2v_txt' in self.space.kwargs_space:
            self._w2v_txt = self.space.kwargs_space['w2v_txt']
        else:
            self._w2v_txt = "NOT_PROVIDED"
        
    def get_w2v(self):
        return self._train_model
    
    def set_w2v(self, value):
        self._train_model = value
        
    def build_naive_representation(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list):

        t1 = time.time()
        print "Starting W2VMatrixHolder representation..."
        
        dimensions = self.dimensions
        
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
            matrix_docs_terms = numpy.zeros((len(corpus_file_list), dimensions),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        mat_terms_dimensions = numpy.zeros((len_vocab, dimensions), dtype=numpy.float64)
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            if term in self._train_model:
                mat_terms_dimensions[u, :] = self._train_model[term]
        self._matrix_terms_dimensions = mat_terms_dimensions
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]

                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        if pal in self._train_model:
                            matrix_docs_terms[i, :] += self._train_model[pal] * freq
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                if tamDoc > 0:
                    matrix_docs_terms[i, :] = matrix_docs_terms[i, :] / tamDoc
                
                i+=1
                
                instance_categories += [autor]
                instance_namefiles += [arch]
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/w2v")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of W2VMatrixHolder representation. Time: ", str(t2-t1)
        
    def build_collect_train_sentences(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list):


        t1 = time.time()
        print "Starting build_collect_train_sentences ..."
        
        self._train_sentences = []
        
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
            #matrix_docs_terms = numpy.zeros((len(corpus_file_list), len_vocab),
            #                            dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                
                self._train_sentences += [tokens]
                
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/w2v")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        #self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of build_collect_train_sentences Time: ", str(t2-t1)
            

class W2VTrainMatrixHolder(W2VMatrixHolder):

    def __init__(self, space, dataset_label="train", train=True):
        super(W2VTrainMatrixHolder, self).__init__(space, w2v=None, dataset_label=dataset_label)
        
        if self._w2v_txt == "NOT_PROVIDED":
            self.build_collect_train_sentences(self.space, 
                                               self.space.virtual_classes_holder_train,
                                               self.space.corpus_file_list_train)
            
            
        if train == True:
            self.train_w2v(self._train_sentences)
            
    def train_w2v(self, sentences):
        
        # print self._train_sentences  
        
        if self._w2v_txt == "NOT_PROVIDED":      
            self._train_model = Word2Vec(self._train_sentences, 
                                         size=self.dimensions, 
                                         min_count=self.min_count, 
                                         workers=self.workers)
        else:
            print "Loading w2v file model ..."
            self._train_model = Word2Vec.load_word2vec_format(self._w2v_txt, binary=False)
            print "End of loading w2v file model ..."   
        
    def build_matrix(self):                
        self.build_naive_representation(self.space,
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
        
    def normalize_matrix(self, normalizer, matrix):
        pass
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix                     
        return self._matrix_terms_dimensions
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_terms_dimensions = value
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
                          
            w2v = self.get_w2v()
            w2v.save(space.space_path + "/w2v/" + space.id_space + "_w2v_model") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
            
            #self.set_matrix_terms(self)   # is this really necessary???
            
        else:
            print "ERROR W2V: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        w2v_train_matrix_holder = W2VTrainMatrixHolder(space, train=False) 
      
        w2v = Word2Vec()
        w2v = w2v.load(space.space_path + "/w2v/" + space.id_space + "_w2v_model")    
        
        w2v_train_matrix_holder.set_w2v(w2v)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        w2v_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        w2v_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        w2v_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return w2v_train_matrix_holder


class W2VTestMatrixHolder(W2VMatrixHolder):

    def __init__(self, 
                 space,
                 train_matrix_holder=None, 
                 dataset_label="test"):
        
        train_matrix_holder = self.load_train_data(space)
        
        super(W2VTestMatrixHolder, self).__init__(space, 
                                                  w2v=train_matrix_holder.get_w2v(),
                                                  dataset_label=dataset_label)     
        
    def build_matrix(self):
        
        self.build_naive_representation(self.space,
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
        
    def normalize_matrix(self, normalizer, matrix):
        pass
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix                     
        return self._matrix_terms_dimensions
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_terms_dimensions = value
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        w2v_train_matrix_holder = W2VTrainMatrixHolder(space, train=False)
              
        w2v = Word2Vec()
        w2v = w2v.load(space.space_path + "/w2v/" + space.id_space + "_w2v_model")    
        
        w2v_train_matrix_holder.set_w2v(w2v)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        w2v_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        w2v_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        w2v_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return w2v_train_matrix_holder
    
    
class VW2VMatrixHolder(MatrixHolder):

    def __init__(self, space, w2v=None, dataset_label="???"):
        super(VW2VMatrixHolder, self).__init__()
        self.space = space
        self.bow_corpus = None    
        self._id_dataset = dataset_label
        self._train_sentences= None
        self._train_model = w2v
        self._matrix_terms_dimensions = None
        
        if ('concepts' in self.space.kwargs_space):        
            self.dimensions = self.space.kwargs_space['concepts']
        else:
            self.dimensions = 100
            
        if ('min_count' in self.space.kwargs_space):        
            self.min_count = self.space.kwargs_space['min_count']
        else:
            self.min_count = 1
            
        if ('workers' in self.space.kwargs_space):        
            self.workers = self.space.kwargs_space['workers']
        else:
            self.workers = 4
            
        if 'w2v_txt' in self.space.kwargs_space:
            self._w2v_txt = self.space.kwargs_space['w2v_txt']
        else:
            self._w2v_txt = "NOT_PROVIDED"
        
    def get_w2v(self):
        return self._train_model
    
    def set_w2v(self, value):
        self._train_model = value
        
    def build_naive_representation(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list):

        t1 = time.time()
        print "Starting BOW representation..."
        
        dimensions = self.dimensions
        
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
            matrix_docs_terms = numpy.zeros((len(corpus_file_list), dimensions),
                                        dtype=numpy.float64)
            dense_flag = True
        
        instance_categories = []
        instance_namefiles = []
        
        ################################################################
        # SUPER SPEED 
        unorder_dict_index = {}
        id2word = {}
        mat_terms_dimensions = numpy.zeros((len_vocab, dimensions), dtype=numpy.float64)
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
            mat_terms_dimensions[u, :] = self._train_model[term]
        self._matrix_terms_dimensions = mat_terms_dimensions
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]

                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        matrix_docs_terms[i, :] += self._train_model[pal] * freq
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                if tamDoc > 0:
                    matrix_docs_terms[i, :] = matrix_docs_terms[i, :] / tamDoc
                
                i+=1
                
                instance_categories += [autor]
                instance_namefiles += [arch]
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/w2v")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)
        
    def build_collect_train_sentences(self,
                          space,
                          virtual_classes_holder,
                          corpus_file_list):


        t1 = time.time()
        print "Starting BOW representation..."
        
        self._train_sentences = []
        
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
        id2word = {}
        for (term, u) in zip(space._vocabulary, range(len_vocab)):
            unorder_dict_index[term] = u
            id2word[u] = term
        ###############################################################    
        
        corpus_bow = []    
        i = 0      
        for autor in space.categories:
            archivos = virtual_classes_holder[autor].cat_file_list
            for arch in archivos:
                tokens = virtual_classes_holder[autor].dic_file_tokens[arch]
                
                self._train_sentences += [tokens]
                
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)
                
                ################################################################
                # SUPER SPEED 
                bow = []
                for pal in docActualFd.keys_sorted():
                    
                    if (pal in unorder_dict_index) and tamDoc > 0:
                        freq = docActualFd[pal] #/ float(tamDoc)
                    else:
                        freq = 0.0
                    
                    if dense_flag:
                        bow += [(unorder_dict_index[pal], freq)]
                        #matrix_docs_terms[i, unorder_dict_index[pal]] = freq
                    
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
                
                corpus_bow += [bow]
            
        Util.create_a_dir(space.space_path + "/w2v")
        
        #print corpus_bow
            
        corpora.MmCorpus.serialize(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm", corpus_bow)
        self.corpus_bow = corpora.MmCorpus(space.space_path + "/w2v/" + space.id_space + "_" + self._id_dataset + "_corpus.mm") # load a corpus of nine documents, from the Tutorials
        
        #print self.corpus_bow
        
        self.id2word = id2word
        
        #self.tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        
        #corpus_tfidf = tfidf[corpus]
        
        #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300, chunksize=1, distributed=True) # run distributed LSA on documents
        #corpus_lsi = lsi[corpus_tfidf]

        self._matrix = matrix_docs_terms
        self._instance_categories = instance_categories
        self._instance_namefiles = instance_namefiles
        
        rows_file.close()
        columns_file.close()
        vals_file.close()

        #print matConceptosTerm

        t2 = time.time()
        print "End of BOW representation. Time: ", str(t2-t1)
            

class VW2VTrainMatrixHolder(W2VMatrixHolder):

    def __init__(self, space, dataset_label="train", train=True):
        super(VW2VTrainMatrixHolder, self).__init__(space, w2v=None, dataset_label=dataset_label)
        self.build_collect_train_sentences(self.space, 
                                           self.space.virtual_classes_holder_train,
                                           self.space.corpus_file_list_train)
        
        if train == True:
            self.train_w2v(self._train_sentences)
            
    def train_w2v(self, sentences):        
        # print self._train_sentences   
        Util.create_a_dir(self.space.space_path + "/w2vC")
        train_path_file = self.space.space_path + "/w2vC/" + self.space.id_space + "train" + self.space.kwargs_space['vectors']
        f_training_file = open(train_path_file, 'w')
        
        for sentence in self._train_sentences:
            f_training_file.write(" ".join(sentence) + "\n")
        
        f_training_file.close()
        
        if self._w2v_txt == "NOT_PROVIDED":
            
            # "time ./word2vec -train textPastorVW.txt -output vectorsPastor.txt -cbow 1 -size 50 -window 4 -negative 25 -hs 0 -sample 0 -threads 1 -binary 0 -iter 3 -min-count 1"
            self._w2v_text = self.space.space_path + "/w2vC/" + self.space.id_space + "vectors.txt" #self.space.kwargs_space['vectors'] 
            #train_path_file = ""
            self._w2v_command = self.space.kwargs_space['exec_path'] + " -train " + train_path_file +  " -output " + self._w2v_text + " -cbow 1 -size " + str(self.space.kwargs_space['concepts']) + " -window 4 -negative 25 -hs 0 -sample 0 -threads 8 -binary 0 -iter 5 -min-count 1"
    
            print self._w2v_command
            subprocess.call(self._w2v_command, shell=True)     
                          
        self._train_model = Word2Vec.load_word2vec_format(self._w2v_text, binary=False)              
        
    def build_matrix(self):                
        self.build_naive_representation(self.space,
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
        
    def normalize_matrix(self, normalizer, matrix):
        pass
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix                     
        return self._matrix_terms_dimensions
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_terms_dimensions = value
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        
        if self is not None:
            cache_file = "%s/%s" % (space.space_path, space.id_space)
            
            #numpy.save(cache_file + "_mat_terms_concepts.npy", 
            #           self.__lsa_train_matrix_holder.get_matrix_terms_concepts())
                          
            w2v = self.get_w2v()
            w2v.save(space.space_path + "/w2v/" + space.id_space + "_w2v_model") # same for tfidf, lda, ...
            
            
            numpy.save(cache_file + "_mat_docs_concepts.npy", 
                       self.get_matrix())
            
            numpy.save(cache_file + "_instance_namefiles.npy", 
                       self.get_instance_namefiles())
            
            numpy.save(cache_file + "_instance_categories.npy", 
                       self.get_instance_categories())
            
            #self.set_matrix_terms(self)   # is this really necessary???
            
        else:
            print "ERROR W2V: There is not a train matrix terms concepts built"

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        w2v_train_matrix_holder = VW2VTrainMatrixHolder(space, train=False) 
      
        w2v = Word2Vec()
        w2v = w2v.load(space.space_path + "/w2v/" + space.id_space + "_w2v_model")    
        
        w2v_train_matrix_holder.set_w2v(w2v)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        w2v_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        w2v_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        w2v_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return w2v_train_matrix_holder


class VW2VTestMatrixHolder(W2VMatrixHolder):

    def __init__(self, 
                 space,
                 train_matrix_holder=None, 
                 dataset_label="test"):
        
        train_matrix_holder = self.load_train_data(space)
        
        super(VW2VTestMatrixHolder, self).__init__(space, 
                                                  w2v=train_matrix_holder.get_w2v(),
                                                  dataset_label=dataset_label)     
        
    def build_matrix(self):
        
        self.build_naive_representation(self.space,
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
        
    def normalize_matrix(self, normalizer, matrix):
        pass
        
    # ------------------------------------------------------        
    def get_matrix_terms(self):    # return some useful information for Decorators e.g. term matrix                     
        return self._matrix_terms_dimensions
    
    def set_matrix_terms(self, value):    # set some useful information for Decorators e.g. term matrix
        self._matrix_terms_dimensions = value
        
    def get_shared_resource(self):    # return some useful information for Decorators e.g. term matrix
        pass
    
    def set_shared_resource(self, value):    # set some useful information for Decorators e.g. term matrix
        pass
    
    def save_train_data(self, space):
        pass

    def load_train_data(self, space):
        cache_file = "%s/%s" % (space.space_path, space.id_space)
        
        w2v_train_matrix_holder = VW2VTrainMatrixHolder(space, train=False)
              
        w2v = Word2Vec()
        w2v = w2v.load(space.space_path + "/w2v/" + space.id_space + "_w2v_model")    
        
        w2v_train_matrix_holder.set_w2v(w2v)
          
        #self.__lsa_train_matrix_holder.set_matrix_terms_concepts(numpy.load(cache_file + "_mat_terms_concepts.npy"))  
        w2v_train_matrix_holder.set_matrix(numpy.load(cache_file + "_mat_docs_concepts.npy"))
        w2v_train_matrix_holder.set_instance_namefiles(numpy.load(cache_file + "_instance_namefiles.npy"))
        w2v_train_matrix_holder.set_instance_categories(numpy.load(cache_file + "_instance_categories.npy"))    
        
        #lsa_train_matrix_holder.set_matrix_terms(lsa_train_matrix_holder)   # this is necessary
        
        return w2v_train_matrix_holder
        
       
        
        
# ----------------------------------------

class ExampleMatrixHolder(MatrixHolder):

    def __init__(self, space):
        super(LSAMatrixHolder, self).__init__()
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
                docActualFd = FreqDistExt(tokens) #virtual_classes_holder[autor].dic_file_fd[arch]
                tamDoc = len(tokens)

                j = 0
                for pal in space._vocabulary:
                    if pal in docActualFd.keys_sorted():
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

class ExampleTrainMatrixHolder(LSAMatrixHolder):

    def __init__(self, space):
        super(LSATrainMatrixHolder, self).__init__(space)

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


class ExampleTestMatrixHolder(BOWMatrixHolder):

    def __init__(self, space):
        super(LSATestMatrixHolder, self).__init__(space)

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
        
# ----------------------------------------        


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
        json.dump(space.get_fdist().keys_sorted(), json_vocabulary)
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
            json.dump(space.get_fdist().keys_sorted(), json_vocabulary)
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
        self._vocabulary = self._fdist.keys_sorted() 
        
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

        self._vocabulary = self._fdist.keys_sorted()
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
                                                   self.categories,
                                                   self)

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
                                                   self.categories,
                                                   self)

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
                                                   self.categories,
                                                   self)

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