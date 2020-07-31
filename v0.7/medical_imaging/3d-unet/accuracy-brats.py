# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

import argparse
import json
import numpy as np
import os
import pickle
import sys
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))

from multiprocessing import Pool
from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions

import time

dtype_map = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="build/logs/mlperf_log_accuracy.json", help="Path to accuracy log json file")
    parser.add_argument("--output_dtype", default="float32", choices=dtype_map.keys(), help="Output data type")
    parser.add_argument("--preprocessed_data_dir", default="build/preprocessed_data", help="Path to the directory containing preprocessed data")
    parser.add_argument("--postprocessed_data_dir", default="build/postprocessed_data", help="Path to the directory containing postprocessed data")
    parser.add_argument("--label_data_dir", default="build/raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr",
        help="Path to the directory containing ground truth labels")
    parser.add_argument("--num_threads_nifti_save", type=int, default=12, help="Number of threads to run the postprocessing with")
    args = parser.parse_args()
    return args

def save_prediction(softmax_mean, output_filename, dct):
    shape_original_before_cropping = dct.get('original_size_of_raw_data')
    bbox = dct.get('crop_bbox')

    seg_old_size = np.zeros(shape_original_before_cropping)
    for c in range(3):
        bbox[c][1] = np.min((bbox[c][0] + softmax_mean.shape[c], shape_original_before_cropping[c]))
    seg_old_size[bbox[0][0]:bbox[0][1],
    bbox[1][0]:bbox[1][1],
    bbox[2][0]:bbox[2][1]] = softmax_mean

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
    seg_resized_itk.SetSpacing(dct['itk_spacing'])
    seg_resized_itk.SetOrigin(dct['itk_origin'])
    seg_resized_itk.SetDirection(dct['itk_direction'])
    sitk.WriteImage(seg_resized_itk, output_filename)

def save_predictions_MLPerf(predictions, output_folder, output_files, dictionaries, num_threads_nifti_save, all_in_gpu, force_separate_z=None, interp_order=3, interp_order_z=0):
    print("Saving predictions...")
    pool = Pool(num_threads_nifti_save)
    results = []
    for i, output_filename in enumerate(output_files):
        print(i, "/", len(output_files))
        output_filename = os.path.join(output_folder, output_filename + ".nii.gz")
        softmax_mean = predictions[i]
        dct = dictionaries[i]
        
        results.append(pool.starmap_async(save_prediction, ((softmax_mean, output_filename, dct),)))

    _ = [i.get() for i in results]

    pool.close()
    pool.join()

    del predictions

def load_loadgen_log(log_file, result_dtype, dictionaries):
    with open(log_file) as f:
        predictions = json.load(f)

    assert len(predictions) == len(dictionaries), "Number of predictions does not match number of samples in validation set!"

    padded_shape = [224, 224, 160]
    results = [None for i in range(len(predictions))]
    for prediction in predictions:
        qsl_idx = prediction["qsl_idx"]
        assert qsl_idx >= 0 and qsl_idx < len(predictions), "Invalid qsl_idx!"
        raw_shape = list(dictionaries[qsl_idx]["size_after_cropping"])
        # Remove the padded part
        pad_before = [(p - r) // 2 for p, r in zip(padded_shape, raw_shape)]
        pad_after = [-(p - r - b) for p, r, b in zip(padded_shape, raw_shape, pad_before)]
        result_shape = tuple(padded_shape)
        result = np.frombuffer(bytes.fromhex(prediction["data"]), result_dtype).reshape(result_shape).astype(np.float16)
        results[qsl_idx] = result[pad_before[0]:pad_after[0], pad_before[1]:pad_after[1], pad_before[2]:pad_after[2]]

    assert all([i is not None for i in results]), "Missing some results!"

    return results

def main():
    args = get_args()
    log_file = args.log_file
    preprocessed_data_dir = args.preprocessed_data_dir
    output_folder = args.postprocessed_data_dir
    ground_truths = args.label_data_dir
    output_dtype = dtype_map[args.output_dtype]
    num_threads_nifti_save = args.num_threads_nifti_save
    all_in_gpu = "None"
    force_separate_z = None
    interp_order = 3
    interp_order_z = 0

    start_time = time.perf_counter()

    # Load necessary metadata.
    print("Loading necessary metadata...")
    with open(os.path.join(preprocessed_data_dir, "preprocessed_files.pkl"), "rb") as f:
        preprocessed_files = pickle.load(f)
    dictionaries = []
    for preprocessed_file in preprocessed_files:
        with open(os.path.join(preprocessed_data_dir, preprocessed_file + ".pkl"), "rb") as f:
            dct = pickle.load(f)[1]
            dictionaries.append(dct)

    # Load predictions from loadgen accuracy log.
    print("Loading loadgen accuracy log...")
    predictions = load_loadgen_log(log_file, output_dtype, dictionaries)

    # Save predictions
    # This runs in multiprocess
    print("Running postprocessing with multiple threads...")
    save_predictions_MLPerf(predictions, output_folder, preprocessed_files, dictionaries, num_threads_nifti_save, all_in_gpu, force_separate_z, interp_order, interp_order_z)

    # Run evaluation
    print("Running evaluation...")
    evaluate_regions(output_folder, ground_truths, get_brats_regions())

    # Load evaluation summary
    print("Loading evaluation summary...")
    with open(os.path.join(output_folder, "summary.csv")) as f:
        for line in f:
            words = line.split(",")
            if words[0] == "mean":
                whole = float(words[1])
                core = float(words[2])
                enhancing = float(words[3])
                mean = (whole + core + enhancing) / 3
                print("Accuracy: mean = {:.5f}, whole tumor = {:.4f}, tumor core = {:.4f}, enhancing tumor = {:.4f}".format(mean, whole, core, enhancing))
                break

    print("Done! Total time: {} sec.".format(time.perf_counter() - start_time))

if __name__ == "__main__":
    main()
