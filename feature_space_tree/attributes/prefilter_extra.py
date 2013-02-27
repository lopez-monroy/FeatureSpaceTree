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

import codecs
from prefilter import DecoratorRawStringNormalizer

class IgnoreStringsDecoratorRawStringNormalizer(DecoratorRawStringNormalizer):
    '''
    This decorator returns an string that ignore Strings in the path_ignored_string 
    file. One use of this filter is when you want to clean a String from some TAGS :)
    '''

    def __init__(self, raw_string_normalizer, path_ignored_strings, to_lower = False):
        super(IgnoreStringsDecoratorRawStringNormalizer, self).__init__(raw_string_normalizer)
        self.path_ignored_strings = path_ignored_strings
        self.to_lower = to_lower

    def get_raw_string(self):

        old_raw_string = self._raw_string_normalizer.get_raw_string()

        f_ignored_strings = codecs.open(self.path_ignored_strings, encoding='utf-8')
        for line in f_ignored_strings:
            if self.to_lower:
                line = line.lower()
            old_raw_string = old_raw_string.replace(line.strip(), "")
            #print line.strip()
        f_ignored_strings.close()

        new_raw_string = old_raw_string

        return new_raw_string
    

