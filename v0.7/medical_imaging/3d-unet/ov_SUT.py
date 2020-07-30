# coding=utf-8
# Copyright (c) 2020 INTEL CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

import time
import array
import json
import os
import sys

import multiprocessing
import concurrent.futures

sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np

from brats_QSL import get_brats_QSL

from openvino.inference_engine import IECore, StatusCode
from scipy.special import softmax

class _3DUNET_OV_SUT():
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        print("Loading OV model...")

        model_xml = model_path
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)

        self.input_name = next(iter(net.inputs))
        self.num_requests = 4

        self.exec_net = ie.load_network(network=net, device_name='CPU', num_requests=self.num_requests)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries_async, self.flush_queries, self.process_latencies)
        self.qsl = get_brats_QSL(preprocessed_data_dir, performance_count)
        print("Finished constructing SUT.")

    @staticmethod
    def output_name():
        return 'output'

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            data = self.qsl.get_features(query_samples[i].index)

            print("Processing sample id {:d} with shape = {:}".format(query_samples[i].index, data.shape))

            before_softmax = self.exec_net.infer(inputs={self.input_name: data[np.newaxis, ...]})[self.output_name].squeeze(0)

            after_softmax = softmax(before_softmax, axis=0)
            output = np.argmax(after_softmax, axis=0).astype(np.float16)

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def issue_queries_async(self, query_samples):
        def _process_output(index, id, output):
            print('Processing output for sample id: {}'.format(index))

            output = softmax(output, axis=0)

            output = np.argmax(output, axis=0).astype(np.float16)

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

        exec_net = self.exec_net

        request_to_sample_id = {}
        outputs = []

        total_requests = len(query_samples)
        idx = 0

        futures = {}

        while total_requests > idx:
            if (total_requests - idx) < len(exec_net.requests):
                streams_to_run = (total_requests - idx)
            else:
                streams_to_run = len(exec_net.requests)

            for infer_request_id in range(0, streams_to_run):
                sample = query_samples[idx + infer_request_id]
                data = self.qsl.get_features(sample.index)

                print("Processing sample id {:d} with shape = {:}".format(sample.index, data.shape))

                infer_request = exec_net.requests[infer_request_id]

                request_to_sample_id[infer_request_id] = {'index': sample.index, 'id': sample.id }

                infer_request.async_infer(inputs={self.input_name: data[np.newaxis, ...]})

            idx += streams_to_run

            exec_net.wait()

            executor = concurrent.futures.ThreadPoolExecutor(streams_to_run)

            for infer_request_id in range(0, streams_to_run):
                infer_request = exec_net.requests[infer_request_id]
                output = infer_request.output_blobs[self.output_name()].buffer.squeeze(0)

                dict = request_to_sample_id[infer_request_id]
                args = (_process_output,
                        dict['index'], dict['id'], output)
                futures[executor.submit(*args)] = infer_request_id
                del request_to_sample_id[infer_request_id]

        concurrent.futures.wait(futures)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass


def get_ov_sut(model_path, preprocessed_data_dir, performance_count):
    return _3DUNET_OV_SUT(model_path, preprocessed_data_dir, performance_count)
