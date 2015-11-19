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
# FeatureSpaceTree:
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

from postfilter import DecoratorByTokenNormalizer

class NeighboringBigramsFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):

    def __init__(self, by_token_normalizer):
        super(NeighboringBigramsFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)

    def get_list_of_tokens(self):
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
        tokens = old_list_of_tokens[2:]  
        
        base_mat = []
        for i in range(rows):
            
            a = i*cols
            b = i*cols + cols
            
            base_mat += [tokens[a:b]]
            
        # print base_mat
        
        if (len(base_mat) != rows) or (len(base_mat[0]) != cols) or (len(base_mat[-1]) != cols):
            print "THE MATRIX HAS A STRANGE SIZE!!!, YOU SHOUL CHECK THIS CASE."
            
        for i in range(rows):
            
            for j in range(cols):
                
                v = ["NOT_A_VISUAL_FEATURE" for e in range(8)]            
                actual = base_mat[i][j]
                
                if ((i-1) >= 0) and ((j-1) >= 0) :                    
                    v[0] = base_mat[i-1][j-1]
                    
                if (j-1) >= 0:
                    v[1] = base_mat[i][j-1]
                    
                if ((i+1) < rows) and ((j-1) >= 0):
                    v[2] = base_mat[i+1][j-1]
                    
                if ((i+1) < rows):
                    v[3] = base_mat[i+1][j]
                    
                if ((i+1) < rows) and ((j+1) < cols):
                    v[4] = base_mat[i+1][j+1]
                    
                if (j+1) < cols:
                    v[5] = base_mat[i][j+1]
                    
                if ((i-1) >= 0) and ((j+1) < cols):
                    v[6] = base_mat[i-1][j+1]
                    
                if (i-1) >= 0:
                    v[7] = base_mat[i-1][j]
                    
                for v_e in v:
                    
                    if v_e != "NOT_A_VISUAL_FEATURE":
                        new_list_of_tokens += [actual + "~" + v_e]
                    
        # print new_list_of_tokens
        return new_list_of_tokens      
    

class OrientationNeighboringBigramsFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):

    def __init__(self, by_token_normalizer):
        super(OrientationNeighboringBigramsFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)

    def get_list_of_tokens(self):
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
        tokens = old_list_of_tokens[2:]  
        
        base_mat = []
        for i in range(rows):
            
            a = i*cols
            b = i*cols + cols
            
            base_mat += [tokens[a:b]]
            
        # print base_mat
        
        if (len(base_mat) != rows) or (len(base_mat[0]) != cols) or (len(base_mat[-1]) != cols):
            print "THE MATRIX HAS A STRANGE SIZE!!!, YOU SHOUL CHECK THIS CASE."
            
        for i in range(rows):
            
            for j in range(cols):
                
                v = ["NOT_A_VISUAL_FEATURE" for e in range(8)]            
                actual = "c" + base_mat[i][j]
                
                if ((i-1) >= 0) and ((j-1) >= 0) :                    
                    v[0] = "no" + base_mat[i-1][j-1]
                    
                if (j-1) >= 0:
                    v[1] = "o" + base_mat[i][j-1]
                    
                if ((i+1) < rows) and ((j-1) >= 0):
                    v[2] = "so" + base_mat[i+1][j-1]
                    
                if ((i+1) < rows):
                    v[3] = "s" + base_mat[i+1][j]
                    
                if ((i+1) < rows) and ((j+1) < cols):
                    v[4] = "se" + base_mat[i+1][j+1]
                    
                if (j+1) < cols:
                    v[5] = "e" + base_mat[i][j+1]
                    
                if ((i-1) >= 0) and ((j+1) < cols):
                    v[6] = "ne" + base_mat[i-1][j+1]
                    
                if (i-1) >= 0:
                    v[7] = "n" + base_mat[i-1][j]
                    
                for v_e in v:
                    
                    if v_e != "NOT_A_VISUAL_FEATURE":
                        new_list_of_tokens += [actual + "~" + v_e]
                    
        # print new_list_of_tokens
        return new_list_of_tokens 
    
    
class NeighboringNoOrderBigramsFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):

    def __init__(self, by_token_normalizer):
        super(NeighboringNoOrderBigramsFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)

    def get_list_of_tokens(self):
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
        tokens = old_list_of_tokens[2:]  
        
        base_mat = []
        for i in range(rows):
            
            a = i*cols
            b = i*cols + cols
            
            base_mat += [tokens[a:b]]
            
        # print base_mat
        
        if (len(base_mat) != rows) or (len(base_mat[0]) != cols) or (len(base_mat[-1]) != cols):
            print "THE MATRIX HAS A STRANGE SIZE!!!, YOU SHOUL CHECK THIS CASE."
            
        for i in range(rows):
            
            for j in range(cols):
                
                v = ["NOT_A_VISUAL_FEATURE" for e in range(8)]            
                actual = base_mat[i][j]
                
                if ((i-1) >= 0) and ((j-1) >= 0) :                    
                    v[0] = base_mat[i-1][j-1]
                    
                if (j-1) >= 0:
                    v[1] = base_mat[i][j-1]
                    
                if ((i+1) < rows) and ((j-1) >= 0):
                    v[2] = base_mat[i+1][j-1]
                    
                if ((i+1) < rows):
                    v[3] = base_mat[i+1][j]
                    
                if ((i+1) < rows) and ((j+1) < cols):
                    v[4] = base_mat[i+1][j+1]
                    
                if (j+1) < cols:
                    v[5] = base_mat[i][j+1]
                    
                if ((i-1) >= 0) and ((j+1) < cols):
                    v[6] = base_mat[i-1][j+1]
                    
                if (i-1) >= 0:
                    v[7] = base_mat[i-1][j]
                    
                for v_e in v:
                    
                    if v_e != "NOT_A_VISUAL_FEATURE":
                        
                        generated_bigram = ""
                        if actual <= v_e:
                            generated_bigram = actual + "~" + v_e
                        else:
                            generated_bigram = v_e + "~" + actual       
                        
                        new_list_of_tokens += [generated_bigram]                    
        # print new_list_of_tokens
        return new_list_of_tokens  
    
    
class W2VNeighboringNoOrderBigramsFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):

    def __init__(self, by_token_normalizer):
        super(W2VNeighboringNoOrderBigramsFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)

    def get_list_of_tokens(self):
        #print "UUUUUUUUUUUUUUUUUUY"
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
        tokens = old_list_of_tokens[2:]  
        
        base_mat = []
        for i in range(rows):
            
            a = i*cols
            b = i*cols + cols
            
            base_mat += [tokens[a:b]]
            
        # print base_mat
        
        if (len(base_mat) != rows) or (len(base_mat[0]) != cols) or (len(base_mat[-1]) != cols):
            print "THE MATRIX HAS A STRANGE SIZE!!!, YOU SHOUL CHECK THIS CASE."
            
        for i in range(rows):
            
            for j in range(cols):
                
                v = ["NOT_A_VISUAL_FEATURE" for e in range(8)]            
                actual = base_mat[i][j]
                
                if ((i-1) >= 0) and ((j-1) >= 0) :                    
                    v[0] = base_mat[i-1][j-1]
                    
                if (j-1) >= 0:
                    v[1] = base_mat[i][j-1]
                    
                if ((i+1) < rows) and ((j-1) >= 0):
                    v[2] = base_mat[i+1][j-1]
                    
                if ((i+1) < rows):
                    v[3] = base_mat[i+1][j]
                    
                if ((i+1) < rows) and ((j+1) < cols):
                    v[4] = base_mat[i+1][j+1]
                    
                if (j+1) < cols:
                    v[5] = base_mat[i][j+1]
                    
                if ((i-1) >= 0) and ((j+1) < cols):
                    v[6] = base_mat[i-1][j+1]
                    
                if (i-1) >= 0:
                    v[7] = base_mat[i-1][j]
                    
                if "NOT_A_VISUAL_FEATURE" in set(v):
                    continue
                
                generated_bigram = []
                cw = 0
                for v_e in v:
                    generated_bigram += [v_e]
                    if cw == 3:
                        generated_bigram += [actual]
                    cw += 1
                    
                new_list_of_tokens += generated_bigram
                #print new_list_of_tokens
        #print "HEEEEEEEEEEEEEEEEEEEEEEY"
        #print new_list_of_tokens
        return new_list_of_tokens 
    
    
class NeighboringNoOrderTrigramsFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):

    def __init__(self, by_token_normalizer):
        super(NeighboringNoOrderTrigramsFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)

    def get_list_of_tokens(self):
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
        tokens = old_list_of_tokens[2:]  
        
        base_mat = []
        for i in range(rows):
            
            a = i*cols
            b = i*cols + cols
            
            base_mat += [tokens[a:b]]
            
        # print base_mat
        
        if (len(base_mat) != rows) or (len(base_mat[0]) != cols) or (len(base_mat[-1]) != cols):
            print "THE MATRIX HAS A STRANGE SIZE!!!, YOU SHOUL CHECK THIS CASE."
            
        for i in range(rows):
            
            for j in range(cols):
                
                v = ["NOT_A_VISUAL_FEATURE" for e in range(8)]
                v_ext = ["NOT_A_VISUAL_FEATURE" for e in range(8)]            
                actual = base_mat[i][j]
                
                if ((i-1) >= 0) and ((j-1) >= 0) and ((i-2) >= 0) and ((j-2) >= 0):                    
                    v[0] = base_mat[i-1][j-1]
                    v_ext[0] = base_mat[i-2][j-2]
                    
                if ((j-1) >= 0) and ((j-2) >= 0):
                    v[1] = base_mat[i][j-1]
                    v_ext[1] = base_mat[i][j-2]
                    
                if (((i+1) < rows) and ((j-1) >= 0)) and (((i+2) < rows) and ((j-2) >= 0)):
                    v[2] = base_mat[i+1][j-1]
                    v_ext[2] = base_mat[i+2][j-2]
                    
                if ((i+1) < rows) and ((i+2) < rows):
                    v[3] = base_mat[i+1][j]
                    v_ext[3] = base_mat[i+2][j]
                    
                if (((i+1) < rows) and ((j+1) < cols)) and (((i+2) < rows) and ((j+2) < cols)):
                    v[4] = base_mat[i+1][j+1]
                    v_ext[4] = base_mat[i+2][j+2]
                    
                if ((j+1) < cols) and ((j+2) < cols):
                    v[5] = base_mat[i][j+1]
                    v_ext[5] = base_mat[i][j+2]
                    
                if (((i-1) >= 0) and ((j+1) < cols)) and (((i-2) >= 0) and ((j+2) < cols)):
                    v[6] = base_mat[i-1][j+1]
                    v_ext[6] = base_mat[i-2][j+2]
                    
                if ((i-1) >= 0) and ((i-2) >= 0):
                    v[7] = base_mat[i-1][j]
                    v_ext[7] = base_mat[i-2][j]
                    
                for v_e, v_e_ext in zip(v, v_ext):
                    
                    if (v_e != "NOT_A_VISUAL_FEATURE") and (v_e_ext != "NOT_A_VISUAL_FEATURE"): 
                        my_seq = [actual, v_e, v_e_ext]
                        
                        if my_seq[0] <= my_seq[-1]:
                            my_seq.reverse()
                            
                        generated_trigram = my_seq[0] + "~" + my_seq[1] + "~" + my_seq[2]                         
                        new_list_of_tokens += [generated_trigram]                    
        # print new_list_of_tokens
        return new_list_of_tokens  
    
#####

class NeighboringNoOrderTetragramsFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):

    def __init__(self, by_token_normalizer):
        super(NeighboringNoOrderTetragramsFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)

    def get_list_of_tokens(self):
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
        tokens = old_list_of_tokens[2:]  
        
        base_mat = []
        for i in range(rows):
            
            a = i*cols
            b = i*cols + cols
            
            base_mat += [tokens[a:b]]
            
        # print base_mat
        
        if (len(base_mat) != rows) or (len(base_mat[0]) != cols) or (len(base_mat[-1]) != cols):
            print "THE MATRIX HAS A STRANGE SIZE!!!, YOU SHOUL CHECK THIS CASE."
            
        for i in range(rows):
            
            for j in range(cols):
                
                v = ["NOT_A_VISUAL_FEATURE" for e in range(8)]
                v_ext = ["NOT_A_VISUAL_FEATURE" for e in range(8)]
                v_ext_2 = ["NOT_A_VISUAL_FEATURE" for e in range(8)]            
                actual = base_mat[i][j]
                
                if ((i-1) >= 0) and ((j-1) >= 0) and ((i-2) >= 0) and ((j-2) >= 0) and ((i-3) >= 0) and ((j-3) >= 0):                    
                    v[0] = base_mat[i-1][j-1]
                    v_ext[0] = base_mat[i-2][j-2]
                    v_ext_2[0] = base_mat[i-3][j-3]
                    
                if ((j-1) >= 0) and ((j-2) >= 0) and ((j-3) >= 0):
                    v[1] = base_mat[i][j-1]
                    v_ext[1] = base_mat[i][j-2]
                    v_ext_2[1] = base_mat[i][j-3]
                    
                if (((i+1) < rows) and ((j-1) >= 0)) and (((i+2) < rows) and ((j-2) >= 0)) and (((i+3) < rows) and ((j-3) >= 0)):
                    v[2] = base_mat[i+1][j-1]
                    v_ext[2] = base_mat[i+2][j-2]
                    v_ext_2[2] = base_mat[i+3][j-3]
                    
                if ((i+1) < rows) and ((i+2) < rows) and ((i+3) < rows):
                    v[3] = base_mat[i+1][j]
                    v_ext[3] = base_mat[i+2][j]
                    v_ext_2[3] = base_mat[i+3][j]
                    
                if (((i+1) < rows) and ((j+1) < cols)) and (((i+2) < rows) and ((j+2) < cols)) and (((i+3) < rows) and ((j+3) < cols)):
                    v[4] = base_mat[i+1][j+1]
                    v_ext[4] = base_mat[i+2][j+2]
                    v_ext_2[4] = base_mat[i+3][j+3]
                    
                if ((j+1) < cols) and ((j+2) < cols) and ((j+3) < cols):
                    v[5] = base_mat[i][j+1]
                    v_ext[5] = base_mat[i][j+2]
                    v_ext_2[5] = base_mat[i][j+3]
                    
                if (((i-1) >= 0) and ((j+1) < cols)) and (((i-2) >= 0) and ((j+2) < cols)) and (((i-3) >= 0) and ((j+3) < cols)):
                    v[6] = base_mat[i-1][j+1]
                    v_ext[6] = base_mat[i-2][j+2]
                    v_ext_2[6] = base_mat[i-3][j+3]
                    
                if ((i-1) >= 0) and ((i-2) >= 0) and ((i-3) >= 0):
                    v[7] = base_mat[i-1][j]
                    v_ext[7] = base_mat[i-2][j]
                    v_ext_2[7] = base_mat[i-3][j]
                    
                for v_e, v_e_ext, v_e_ext_2 in zip(v, v_ext, v_ext_2):
                    
                    if (v_e != "NOT_A_VISUAL_FEATURE") and (v_e_ext != "NOT_A_VISUAL_FEATURE") and (v_e_ext_2 != "NOT_A_VISUAL_FEATURE"): 
                        my_seq = [actual, v_e, v_e_ext, v_e_ext_2]
                        
                        if my_seq[0] <= my_seq[-1]:
                            my_seq.reverse()
                            
                        generated_tetragram = my_seq[0] + "~" + my_seq[1] + "~" + my_seq[2] + "~" + my_seq[3]                         
                        new_list_of_tokens += [generated_tetragram]                    
        # print new_list_of_tokens
        return new_list_of_tokens  
    
#####

class SkipSizeInfoFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):

    def __init__(self, by_token_normalizer):
        super(SkipSizeInfoFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)

    def get_list_of_tokens(self):
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
                     
        new_list_of_tokens = old_list_of_tokens[2:]
        return new_list_of_tokens
    

class MaxPatternsByRowFilterDecoratorByTokenNormalizer(DecoratorByTokenNormalizer):
    
    max_patterns = None

    def __init__(self, by_token_normalizer, path_to_max_patterns, a, b):
        super(MaxPatternsByRowFilterDecoratorByTokenNormalizer, self).__init__(by_token_normalizer)
        self.path_to_max_patterns = path_to_max_patterns
        self.a = a
        self.b = b
        
    def count_max_patterns_in_row(self, max_pattern, row):
                
        count = 0
        
        if len(set(max_pattern) & set(row)) == len(set(max_pattern)):
        
            index_max_pattern = 0
            for elem_row in row:
                
                if elem_row == max_pattern[index_max_pattern]:
                    
                    index_max_pattern += 1
                    
                    # if you have found the last element of the pattern, then restart the index and count it.
                    if index_max_pattern == len(max_pattern):
                        index_max_pattern = 0
                        count += 1
                    
        return count   

    def get_list_of_tokens(self):
        old_list_of_tokens = self._by_token_normalizer.get_list_of_tokens()
        # print old_list_of_tokens
        
        # Read the list of max patterns ----------------------------------------
        
        max_patterns = []
        if MaxPatternsByRowFilterDecoratorByTokenNormalizer.max_patterns == None:
            f_of_max_patterns = open(self.path_to_max_patterns, "r")
            
            for line in f_of_max_patterns:
                
                elements = line.split()
                
                if int(elements[1]) >= self.a and int(elements[1]) <= self.b:
                    max_patterns += [elements[2:]] 
            
            print "MAX_PATTERNS: ", max_patterns
            MaxPatternsByRowFilterDecoratorByTokenNormalizer.max_patterns = max_patterns
            f_of_max_patterns.close()
        else:
            max_patterns = MaxPatternsByRowFilterDecoratorByTokenNormalizer.max_patterns
                
        # ----------------------------------------------------------------------
        
        new_list_of_tokens = []
        
        rows = int(old_list_of_tokens[0])
        cols = int(old_list_of_tokens[1])
        tokens = old_list_of_tokens[2:]  
        
        base_mat = []
        for i in range(rows):
            
            a = i*cols
            b = i*cols + cols
            
            base_mat += [tokens[a:b]]
            
        # print base_mat
        
        if (len(base_mat) != rows) or (len(base_mat[0]) != cols) or (len(base_mat[-1]) != cols):
            print "THE MATRIX HAS A STRANGE SIZE!!!, YOU SHOULD CHECK THIS CASE."
            
        for i in range(rows):
            
            for max_pattern in max_patterns:
                freq = 0
                freq = self.count_max_patterns_in_row(max_pattern, base_mat[i])
                
                tag_pattern = "~".join(max_pattern)
                
                new_list_of_tokens += [tag_pattern] * freq
                
        # print new_list_of_tokens
        return new_list_of_tokens 