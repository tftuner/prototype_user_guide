import os
import nni
import time
import signal
import subprocess
import shlex
import psutil
import cuhk_prototype_tuner_v2
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

logging.info("Current trial using GPU index %s" % os.environ["CUDA_VISIBLE_DEVICES"])

def kill_process_and_children(proc_pid):
  process = psutil.Process(proc_pid)
  for proc in process.children(recursive=True):
      proc.kill()
  process.kill()

params = {
  'dropout':0.0,
  'label_smooth': 0.1,
  'lr':0.00001,
  'lr_scheduler':"inverse_sqrt",
  'warmup_update':2500,
  'optimizer':"adam",
  'inter_op_parallelism_threads':1,
  'intra_op_parallelism_threads':2,
  'benchmark':0,
  'allow_tf32':0,
} 

tuned_params = nni.get_next_parameter() 
params.update(tuned_params) 
t_id = nni.get_trial_id()
e_id = nni.get_experiment_id()


train_batch_size = 1
path_2_data="/research/d3/zmwu/model/mbart_company_version/post_process/en-zh_100/"
lang_pairs="en_XX-zh_CN,zh_CN-en_XX"
lang_list="/research/d3/zmwu/model/mbart_company_version/lang_list"
pretrained_model="/research/d3/zmwu/model/mbart_company_version/mbart.cc25/model.pt"
user_dir="/research/d3/zmwu/model/mbart_company_version/mbart"
save_dir="/research/d3/zmwu/model/mbart_company_version/ckpt"

is_load,load_path,save_path,budget = cuhk_prototype_tuner_v2.preprocess(t_id,params,save_dir)

if not os.path.exists(save_path):
    os.makedirs(save_path)


if is_load:
  logging.info("Load previous training result from %s" % load_path)
  load_file = os.path.join(load_path,"checkpoint_last.pt")
  s_time = time.time()

  train_cmd = "fairseq-train %s --user-dir %s --save-dir %s --encoder-normalize-before --decoder-normalize-before --arch mbart_large --layernorm-embedding --task translation_multi_simple_epoch_nni --sampling-method \"temperature\" --sampling-temperature 1.5 --encoder-langtok \"src\" --decoder-langtok --lang-dict \"%s\" --lang-pairs \"%s\" --criterion label_smoothed_cross_entropy --min-lr -1 --empty-cache-freq 4 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 2048 --update-freq 4 --fp16 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 10 --dropout %f --label-smoothing %f --lr %f  --lr-scheduler %s --warmup-updates %d --optimizer %s --inter %d --intra %d --benchmark %d --allow_tf32 %d --restore-file %s --max-epoch %d --save-interval %d --batch-size %d "%(path_2_data,user_dir,save_path,lang_list,lang_pairs,params['dropout'],params['label_smooth'],params['lr'],params['lr_scheduler'],params['warmup_update'],params['optimizer'],int(params['inter_op_parallelism_threads']),int(params['intra_op_parallelism_threads']),int(params['benchmark']),int(params['allow_tf32']),load_file,params['TRIAL_BUDGET'],params['TRIAL_BUDGET'],train_batch_size)

  train_process = subprocess.Popen(shlex.split(train_cmd),stdout=subprocess.PIPE,shell=False,bufsize=1)
  train_pid = train_process.pid
  logging.info("train process start,process ID is %d" % train_pid)
  for stdout_line in iter(train_process.stdout.readline, b""):
    print(stdout_line.decode(),end='')
  train_process.stdout.close()
  train_process.wait()
  logging.info('train process finish, check if train process close properly...')
  if psutil.pid_exists(train_pid):
    logging.info("trian process still exists, kill it.")
    kill_process_and_children(train_pid)
  else:
    logging.info("train process finish and exit normally.")
  
  e_time = time.time()
  shutil.rmtree(load_path)
else:
  logging.info("No pervious trianing found.")
  s_time = time.time()

  train_cmd = "fairseq-train %s --user-dir %s --save-dir %s --finetune-from-model %s --encoder-normalize-before --decoder-normalize-before --arch mbart_large --layernorm-embedding --task translation_multi_simple_epoch_nni --sampling-method \"temperature\" --sampling-temperature 1.5 --encoder-langtok \"src\" --decoder-langtok --lang-dict \"%s\" --lang-pairs \"%s\" --criterion label_smoothed_cross_entropy --min-lr -1   --empty-cache-freq 4 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 2048 --update-freq 4 --fp16  --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 10 --dropout %f --label-smoothing %f --lr %f  --lr-scheduler %s --warmup-updates %d --optimizer %s --inter %d --intra %d --benchmark %d --allow_tf32 %d  --max-epoch %d --save-interval %d --batch-size %d "%(path_2_data,user_dir,save_path,pretrained_model,lang_list,lang_pairs,params['dropout'],params['label_smooth'],params['lr'],params['lr_scheduler'],params['warmup_update'],params['optimizer'],int(params['inter_op_parallelism_threads']),int(params['intra_op_parallelism_threads']),int(params['benchmark']),int(params['allow_tf32']),budget,budget,train_batch_size)

  train_process = subprocess.Popen(shlex.split(train_cmd),stdout=subprocess.PIPE,shell=False,bufsize=1)
  train_pid = train_process.pid
  logging.info("train process start,process ID is %d" % train_pid)
  for stdout_line in iter(train_process.stdout.readline, b""):
    print(stdout_line.decode(),end='')
  train_process.stdout.close()
  train_process.wait()
  logging.info('train process finish, check if train process close properly...')
  if psutil.pid_exists(train_pid):
    logging.info("trian process still exists, kill it.")
    kill_process_and_children(train_pid)
  else:
    logging.info("train process finish and exit normally.")
  
  e_time = time.time()


spent_time = (e_time - s_time) / 3600.0
logging.info("time spenting on training: %fh" % spent_time)


ckpt_path = os.path.join(save_path,"checkpoint_last.pt")
gen_subset = "test"
spm = "/research/d3/zmwu/model/mbart/mbart.cc25/sentence.bpe.model"


lang_src = ["en_XX","zh_CN"]
lang_tgt = ["zh_CN","en_XX"]
GPU_list = os.environ["CUDA_VISIBLE_DEVICES"].strip().split(",")

# key = GPU index(char) ; value = generate process obj
generate_process_dict = {}
bleu_score_list = []

assert(len(lang_tgt) == len(lang_src),"number of src language must equal to number of taget language!")


## create generate process for each language and assign 1 GPU to run it.
for i in range(len(lang_src)):
  dir_name = lang_src[i] + "_to_" + lang_tgt[i]
  generate_process_output_dir = os.path.join(save_path,dir_name)

  if not os.path.exists(generate_process_output_dir):
    os.makedirs(generate_process_output_dir)


  if len(generate_process_dict) < len(GPU_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list[i]
    logging.info("using GPU:%s"%os.environ["CUDA_VISIBLE_DEVICES"])

    generate_cmd = "fairseq-generate --path=%s %s --user-dir %s --task translation_multi_simple_epoch_nni --encoder-langtok 'src' --decoder-langtok --gen-subset %s -s %s -t %s --lang-dict %s --lang-pairs %s --bpe 'sentencepiece' --empty-cache-freq 1 --sentencepiece-model %s --scoring 'sacrebleu' --fp16 --max-sentences 128 --results-path %s"%(ckpt_path,path_2_data,user_dir,gen_subset,lang_src[i],lang_tgt[i],lang_list,lang_pairs,spm,generate_process_output_dir)

    generate_process = subprocess.Popen(shlex.split(generate_cmd),shell=False,bufsize=1)
    logging.info("generate process start,process ID is %d" % generate_process.pid)
    generate_process_dict[GPU_list[i]] = generate_process
  
  else:
    logging.info("all GPUs are assigned tasks, wait for idle GPU.")

    waitting = True
    free_gpu_index = None

    while waitting:
      for g in GPU_list:
        proc_state = subprocess.Popen.poll(generate_process_dict[g])
        if proc_state == 0:
            logging.info('generate process %d finish, check if generate process close properly...' % generate_process_dict[g].pid)
            if psutil.pid_exists(generate_process_dict[g].pid):
              logging.info("generate process %d still exists, kill it." % generate_process_dict[g].pid)
              kill_process_and_children(generate_process_dict[g].pid)
            else:
              logging.info("generate process %d finish and exit normally." % generate_process_dict[g].pid)
            free_gpu_index = g
            waitting = False
            break
      if waitting == True:
        logging.info("no idle GPU,sleep 10")
        time.sleep(10)
      else:
        logging.info("find idle GPU %s" % free_gpu_index)


    os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu_index
    logging.info("using GPU:%s"%os.environ["CUDA_VISIBLE_DEVICES"])

    generate_cmd = "fairseq-generate --path=%s %s --user-dir %s --task translation_multi_simple_epoch_nni --encoder-langtok 'src' --decoder-langtok --gen-subset %s -s %s -t %s --lang-dict %s --lang-pairs %s --bpe 'sentencepiece' --empty-cache-freq 1 --sentencepiece-model %s --scoring 'sacrebleu' --fp16 --max-sentences 128 --results-path %s"%(ckpt_path,path_2_data,user_dir,gen_subset,lang_src[i],lang_tgt[i],lang_list,lang_pairs,spm,generate_process_output_dir)

    generate_process = subprocess.Popen(shlex.split(generate_cmd),shell=False,bufsize=1)
    logging.info("generate process start,process ID is %d" % generate_process.pid)
    generate_process_dict[free_gpu_index] = generate_process



## wait for all generate process finish.
for g in GPU_list:
  generate_process_dict[g].wait()
  logging.info('generate process %d finish, check if generate process close properly...' % generate_process_dict[g].pid)
  if psutil.pid_exists(generate_process_dict[g].pid):
    logging.info("generate process %d still exists, kill it." % generate_process_dict[g].pid)
    kill_process_and_children(generate_process_dict[g].pid)
  else:
    logging.info("generate process %d finish and exit normally." % generate_process_dict[g].pid)


## calculate bleu score for each language pair
for i in range(len(lang_src)):
  dir_name = lang_src[i] + "_to_" + lang_tgt[i]
  generate_process_output_dir = os.path.join(save_path,dir_name)

  target_file = os.path.join(generate_process_output_dir,"ref_file")
  result_file = os.path.join(generate_process_output_dir,"generate-test.txt")
  
  os.system("cat %s | grep -P \"^T\" | cut -f 2- > %s" % (result_file, target_file))
  bsf = os.popen("cat %s | grep -P \"^D\" | cut -f 3- | sacrebleu -b --tok zh %s" % (result_file,target_file))
  bs = float(bsf.readline())
  logging.info("bleu score of %s to %s : %d" % (lang_src[i],lang_tgt[i],bs))
  bleu_score_list.append(bs)

bs_sum = 0.0
for s in bleu_score_list:
  bs_sum = bs_sum + s
final_bs = bs_sum / len(bleu_score_list)
report_dict = {'runtime':spent_time,'default':final_bs,'maximize':['default']}
is_save = cuhk_prototype_tuner_v2.postprocess(
  e_id, 
  t_id, 
  report_dict,
  params,
  os.path.dirname(save_path)
)
if not is_save:
  try:
    logging.info(f"Don't save the current trial. Try to remove dir:{save_path}")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
  except OSError as e:
    print("Error: %s : %s" % (save_path, e.strerror))
nni.report_final_result(report_dict)
