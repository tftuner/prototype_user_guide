import argparse
import numpy as np
import sys
import os

from bilm.training import train, load_options_latest_checkpoint, load_vocab, test
from bilm.data import BidirectionalLMDataset

import warnings
import cuhk_prototype_tuner_v2
import shutil

warnings.filterwarnings('ignore')

import nni  # NNI modification
import time  # NNI modification
import tensorflow as tf # NNI modification


def main(args):
    is_load,load_path,save_path,budget = cuhk_prototype_tuner_v2.preprocess(t_id,params,args.save_dir)

    vocab = load_vocab(args.vocab_file, 50)

    batch_size = int(params['batch_size'])

    gpus_index_list = list(map(int,os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    n_train_tokens = 768648884

    sess_config = tf.compat.v1.ConfigProto( 
                    allow_soft_placement=True,
                    inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
                    intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
                    graph_options=tf.compat.v1.GraphOptions(
                      infer_shapes=params['infer_shapes'],
                      place_pruned_graph=params['place_pruned_graph'],
                      enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
                      optimizer_options=tf.compat.v1.OptimizerOptions(
                        do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
                        max_folded_constant_in_bytes=int(params['max_folded_constant']),
                        do_function_inlining=params['do_function_inlining'],
                        global_jit_level=params['global_jit_level'])))

    options = {
     'bidirectional': True,
     'char_cnn': {'activation': 'relu','embedding': {'dim': 16},
              'filters': [[1, 32],[2, 32],[3, 64],[4, 128],[5, 256],[6, 512],[7, 1024]],
              'max_characters_per_token': 50,
              'n_characters': 261,
              'n_highway': 2},
     'dropout': 0.1,
     'lstm': {'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},
     'all_clip_norm_val': 10.0,
     'n_epochs': int(budget),  # NNI modification
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }
    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,shuffle_on_load=True)
    tf_save_dir = save_path
    tf_log_dir = save_path
    if not os.path.exists(tf_save_dir) :
      os.makedirs(tf_save_dir)

    if params['tf_gpu_thread_mode'] in ["global", "gpu_private", "gpu_shared"]:
        os.environ['TF_GPU_THREAD_MODE'] = params['tf_gpu_thread_mode']
    if is_load:
      load_file = os.path.join(load_path,'model.ckpt')
      start = time.time()
      final_perplexity = train(options, data, n_gpus,gpus_index_list,tf_save_dir, tf_log_dir,sess_config,restart_ckpt_file=load_file)  
      end = time.time()
      shutil.rmtree(load_path)
    else:
      start = time.time()
      final_perplexity = train(options, data, n_gpus,gpus_index_list,tf_save_dir, tf_log_dir,sess_config)
      end = time.time()
    spent_time = (end - start) / 3600.0
    if args.test_prefix != '':
      options, ckpt_file = load_options_latest_checkpoint(tf_save_dir)
      kwargs = {
          'test': True,
          'shuffle_on_load': False,
      }
      test_data = BidirectionalLMDataset(args.test_prefix, vocab, **kwargs)
      final_perplexity = test(options, ckpt_file, test_data, batch_size=128)
    report_dict = {'runtime':spent_time,'default':final_perplexity}   
    nni.report_final_result(report_dict)
  ### NNI modification ###
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--test_prefix', help='Prefix for test files', default='')
    args = parser.parse_args()

    ### NNI modification ###
    params = {
      'batch_size': 128,
      'inter_op_parallelism_threads':1,
      'intra_op_parallelism_threads':2,
      'infer_shapes':0,
      'place_pruned_graph':0,
      'enable_bfloat16_sendrecv':0,
      'do_common_subexpression_elimination':0,
      'max_folded_constant':2,
      'do_function_inlining':0,
      'global_jit_level':1,
      'tf_gpu_thread_mode':"global"
    }
    tuned_params = nni.get_next_parameter() 
    params.update(tuned_params)
    t_id = nni.get_trial_id() 
    ### NNI modification ###

    main(args)
