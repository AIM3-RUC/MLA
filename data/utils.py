# Copyright(c) 2022 Liang Zhang 
# E-Mail: <zhangliang00@ruc.edu.cn>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lmdb
import numpy as np
import msgpack, msgpack_numpy
msgpack_numpy.patch()


class LMDBReader():
    def __init__(self, lmdb_path, pack='msgpack'):
        self.env = lmdb.open(lmdb_path, readonly=True, create=False, readahead=False)
        self.txn = self.env.begin(write=False, buffers=True)
        self.pack = pack

    def __getitem__(self, key):
        dump = self.txn.get(key.encode('utf-8'))
        if self.pack == 'msgpack':
            dump = msgpack.loads(dump, raw=False)
            return dump
        else:
            raise NotImplementedError

    def __del__(self):
        self.env.close()


class LMDBRawImageReader(LMDBReader):
    pass

class DetectFeatReader():
    def __init__(self, lmdb_path):
        self.reader = LMDBReader(lmdb_path, 'msgpack')
    
    def __getitem__(self, key):
        if key[-4:] != '.npz':
            key = key.split('.')[0]
            key = key + '.npz'
        
        img_dump = self.reader[key]
        bboxes = img_dump['norm_bb']
        features = img_dump['features']
        nbbs = len(bboxes)

        return features, bboxes, nbbs