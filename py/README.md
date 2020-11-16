# Manual for CUHKPrototypeTunerV2

## Install CUHKPrototypeTunerV2

1. Run the following command 

   ```bash
   pip install nni && \
   wget https://github.com/vincentcheny/hpo-training/releases/download/cuhk_prototype_tuner_v2_2.1/CUHKPrototypeTunerV2-2.1-py3-none-any.whl && \
   nnictl package install CUHKPrototypeTunerV2-2.1-py3-none-any.whl
   ```

2. if success install, you should see this output  in the command line

   ```bash
   Installing collected packages: CUHKPrototypeTunerV2
   Successfully installed CUHKPrototypeTunerV2-2.1
   CUHKPrototypeTunerV2 installed!
   ```

## For ELMO 

1. Clone the source code of ELMO from git 

   ```bash
   git clone https://github.com/allenai/bilm-tf.git
   ```

2. Go to the working directory

   ```bash
   cd bilm-tf
   ```

3. Create file  ``search_space.json`` to define the search space of hyperparameters and hardware parameters. Execute: 

   ```json
   cat << EOF > search_space.json
   {
       "epoch":{"_type": "uniform", "_value": [5, 50]},
       "batch_size":{"_type": "uniform", "_value": [64, 512]},
       "inter_op_parallelism_threads":{"_type":"choice","_value":[1,2,3,4]},
       "intra_op_parallelism_threads":{"_type":"choice","_value":[2,4,6,8,10,12]},
       "infer_shapes":{"_type":"choice","_value":[0,1]},
       "place_pruned_graph":{"_type":"choice","_value":[0,1]},
       "enable_bfloat16_sendrecv":{"_type":"choice","_value":[0,1]},
       "do_common_subexpression_elimination":{"_type":"choice","_value":[0,1]},
       "max_folded_constant":{"_type":"choice","_value":[2,4,6,8,10]},
       "do_function_inlining":{"_type":"choice","_value":[0,1]},
       "global_jit_level":{"_type":"choice","_value":[0,1,2]},
       "tf_gpu_thread_mode":{"_type":"choice","_value":["global", "gpu_private", "gpu_shared"]}
   }
   EOF
   ```

4. Create file  ``config.yml`` with following content. Execute:

   ```yml
   cat << EOF > config.yml
   authorName: lscm
   experimentName: elmo
   trialConcurrency: 1
   
   # Specify the maximum runing time, we specify 1 week here
   maxExecDuration: 40h 
   maxTrialNum: 9999
   trainingServicePlatform: local
   searchSpacePath: search_space.json
   useAnnotation: false
   advisor:
     builtinAdvisorName: CUHKPrototypeTunerV2
     classArgs:
       num_epochs: 25
   trial:
     command: python ./bin/train_elmo.py --train_prefix=./data/one_billion/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* --vocab_file ./data/vocab-2016-09-10.txt --save_dir ./output_model
     codeDir: .
     # specifiy the number of GPU used by each trial
     gpuNum: 4
   localConfig:
     # specific index of GPU used by the experiment, just like CUDA_VISIBLE_DEVICES
     gpuIndices: "0,1,2,3"
   EOF
   ```
   
5. Replace `bilm/training.py` and `train_elmo.py` to apply configuration from tuner and report performance metrics

```bash
wget https://raw.githubusercontent.com/wuzhuoming/CUHKPrototypeTunerV2/fix_elmo/training.py -O bilm/training.py && \
wget https://raw.githubusercontent.com/wuzhuoming/CUHKPrototypeTunerV2/fix_elmo/train_elmo.py -O bin/train_elmo.py 
```

6. The tuning is ready to [start](#start-tuning) 

## For mBART 

1. Create a working directory "mbart"

   ```bash
   mkdir mbart && cd mbart
   ```

2. Create user directory and download customized task file.

   ```bash
   mkdir user_dir
   
   wget https://raw.githubusercontent.com/wuzhuoming/CUHKPrototypeTunerV2/fix_elmo/__init__.py -O user_dir/__init__.py
   
   wget https://raw.githubusercontent.com/wuzhuoming/CUHKPrototypeTunerV2/fix_elmo/translation_multi_simple_epoch_nni.py -O user_dir/translation_multi_simple_epoch_nni.py
   ```

3. Create file  ``search_space.json`` to define the search space of hyperparameters and hardware parameters. Execute: 

   ```json
   cat << EOF > search_space.json
   {
       "dropout":{"_type":"choice","_value":[0.0,0.1,0.2,0.3]},
       "label_smooth":{"_type":"choice","_value":[0.1,0.2,0.3]},
       "lr":{"_type":"choice","_value":[0.00001,0.00002,0.00003,0.00004,0.00005,0.00006]},
       "lr_scheduler":{"_type":"choice","_value":["inverse_sqrt","polynomial_decay"]},
       "warmup_update":{"_type":"choice","_value":[2500,3000,3500,4000,4500,5000]},
       "optimizer":{"_type":"choice","_value":["adagrad", "adam"]},
   
       "inter_op_parallelism_threads":{"_type":"choice","_value":[1,2,3,4]},
       "intra_op_parallelism_threads":{"_type":"choice","_value":[2,4,6,8,10,12]},
       "benchmark":{"_type":"choice","_value":[0,1]},
       "allow_tf32":{"_type":"choice","_value":[0,1]}
   }
   EOF
   ```

4. Create file  ``config.yml`` with following content. Execute:

   ```yml
   cat << EOF > config.yml
   authorName: lscm
   experimentName: MBART
   trialConcurrency: 1
   
   # Specific the maximum runing time, we specific 1 week here
   maxExecDuration: 40h 
   maxTrialNum: 9999
   trainingServicePlatform: local
   searchSpacePath: search_space.json
   useAnnotation: false
   advisor:
     builtinAdvisorName: CUHKPrototypeTunerV2
     classArgs:
       num_epochs: 25
   trial:
     command: python wrap_program_mbart.py
     codeDir: .
     # specifiy the number of GPU used by each trial
     gpuNum: 4
   localConfig:
     # specifiy index of GPU used by the experiment, just like CUDA_VISIBLE_DEVICES
     gpuIndices: "0,1,2,3"
   EOF
   ```
   
5. Download the tuner program "wrap_program_mbart.py"

   ```bash
   wget https://raw.githubusercontent.com/wuzhuoming/CUHKPrototypeTunerV2/fix_elmo/wrap_program_mbart.py
   ```

6. The tuning is ready to [start](#start-tuning).


### For MASS 

1. Clone the source code of MASS from git 

   ```bash
   git clone https://github.com/microsoft/MASS.git
   ```

2. Go to the working dir

   ```bash
   cd MASS/MASS-supNMT
   ```

3. Create file  "search_space.json" to define the search space of hyperparameters and hardware parameters. Execute: 

   ```json
   cat << EOF > search_space.json
   {
       "clip-norm":{"_type":"choice","_value":[0.0,0.1,0.2,0.3,0.4,0.5]},
       "lr":{"_type":"choice","_value":[0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]},
       "dropout":{"_type":"choice","_value":[0.1,0.2,0.3,0.4,0.5]},
       "relu-dropout":{"_type":"choice","_value":[0.1,0.2,0.3,0.4,0.5]},
       "attention-dropout":{"_type":"choice","_value":[0.1,0.2,0.3,0.4,0.5]},
       "optimizer":{"_type":"choice","_value":["adagrad", "adam"]},
       
   
       "inter_op_parallelism_threads":{"_type":"choice","_value":[1,2,3,4]},
       "intra_op_parallelism_threads":{"_type":"choice","_value":[2,4,6,8,10,12]},
       "benchmark":{"_type":"choice","_value":[0,1]},
       "allow_tf32":{"_type":"choice","_value":[0,1]}
   }
   EOF
   ```

4. Create file  "config.yml" with following content. Execute:

   ```yml
   cat << EOF > config.yml
   authorName: lscm
   experimentName: MASS
   trialConcurrency: 1
   
   # Specific the maximum runing time, we specific 1 week here
   maxExecDuration: 40h 
   maxTrialNum: 9999
   trainingServicePlatform: local
   searchSpacePath: search_space.json
   useAnnotation: false
   advisor:
     builtinAdvisorName: CUHKPrototypeTunerV2
     classArgs:
       num_epochs: 25
   trial:
     command: python wrap_program_mass.py
     codeDir: .
     # specifiy the number of GPU used by each trial
     gpuNum: 4
   localConfig:
     # specifiy index of GPU used by the experiment, just like CUDA_VISIBLE_DEVICES
     gpuIndices: "0,1,2,3"
   EOF
   ```
   
5. Download file "wrap_program_mass.py" in the same directory of "config.yml".

   ```bash
   wget https://raw.githubusercontent.com/wuzhuoming/CUHKPrototypeTunerV2/fix_elmo/wrap_program_mass.py
   ```

6. Replace `mass/xmasked_seq2seq.py` to apply configuration from tuner and report performance metrics

```bash
wget https://raw.githubusercontent.com/wuzhuoming/CUHKPrototypeTunerV2/fix_elmo/xmasked_seq2seq.py -O mass/xmasked_seq2seq.py
```

6. The tuning is ready to [start](#start-tuning) 


## Start tuning 

1. Inside the working directory, execute 

   ```bash
   nnictl create --config ./config.yml -p 8080
   ```

2. If successfully start the tuner with training program, you should see the following message in the console:
   Your experiment id `egchD4qy` is shown. Please note it down for [pausing the tuning](#stop-and-resume-tuning) later on. 

    ```log
    INFO: Starting restful server...
    INFO: Successfully started Restful server!
    INFO: Setting local config...
    INFO: Successfully set local config!
    INFO: Starting experiment...
    INFO: Successfully started experiment!
    -----------------------------------------------------------------------
    The experiment id is egchD4qy
    The Web UI urls are: [Your IP]:8080
    -----------------------------------------------------------------------
    
    You can use these commands to get more information about the experiment
    -----------------------------------------------------------------------
             commands                       description
    1. nnictl experiment show        show the information of experiments
    2. nnictl trial ls               list all of trial jobs
    3. nnictl top                    monitor the status of running experiments
    4. nnictl log stderr             show stderr log content
    5. nnictl log stdout             show stdout log content
    6. nnictl stop                   stop an experiment
    7. nnictl trial kill             kill a trial job by id
    8. nnictl --help                 get help information about nnictl
    -----------------------------------------------------------------------
    ```

3. To check Tuning Process From Web UI, open a browser and use the URL given by NNI:

```
The Web UI urls are: [Your IP]:8080
```

#### **Overview page**

Information about this experiment will be shown in the WebUI, including the experiment trial profile and search space message. NNI also supports downloading this information and the parameters through the **Download** button. You can download the experiment results during or after the execution.

<img src="https://lh3.googleusercontent.com/-LzY00M6Qj5s/X1r3WeZybeI/AAAAAAAAAdc/rDSdoBefp5oDhz3XKMHBxbTL3zr1ENHcACK8BGAsYHg/s0/2020-09-10.png"/>

The top 10 trials will be listed on the Overview page. You can browse all the trials on the “Trials Detail” page.

<img src="https://lh3.googleusercontent.com/-nUV6DIuojdw/X1r3Fzve2TI/AAAAAAAAAdU/SYO259Bld24Lzhvsn7UfJu8XT7NGnpOBQCK8BGAsYHg/s0/2020-09-10.png"/>

#### **Trials Details page**

Click the “Default Metric” tab to see the point graph of all trials. Hover to see specific default metrics and search space messages.

<img src="https://lh3.googleusercontent.com/-U9ZPPo3mY1s/X1r4Uu9cLYI/AAAAAAAAAdk/GCutOTm5FDQlLJ-HVyriXuphdU2BJRwtgCK8BGAsYHg/s0/2020-09-10.png"/>

Click the “Hyper Parameter” tab to see the parallel graph.

- You can select the percentage to see the top trials.

- Choose two axis to swap their positions.

  <img src="https://nni.readthedocs.io/en/latest/_images/hyperPara.png"/>

  

  Below is the status of all trials. Specifically:

  - Trial detail: trial’s id, duration, status, accuracy, and search space file.

  - If you run on the OpenPAI platform, you can also see the hdfsLogPath.

  - Kill: you can kill a job that has the `Running` status.

  - Support: Used to search for a specific trial.

    <img src="https://lh3.googleusercontent.com/-zGe44tn04GY/X1r4y8KRZRI/AAAAAAAAAds/0VCQ0B3eQbcbKMM323WQdKGz9xPQTJrdgCK8BGAsYHg/s0/2020-09-10.png" />



## Stop And Resume tuning

To stop current experiment, you can use this command:

```bash
nnictl stop egchD4qy # egchD4qy is the experiment id provided by the tuner when you launch it 
```

To resume the experiment you stop:

```bash
nnictl resume egchD4qy
```

## Get the trained model

1. The tuner execute the user training program multiple times (we call it `trial`) with different configurations.
   There are multiple trained models generated throughout the whole tuning. 

2. To obtain the trained model with highest accuracy, go to the `Overview Page` and check the top 10 trials with best accuracy 
   <img src="https://lh3.googleusercontent.com/-nUV6DIuojdw/X1r3Fzve2TI/AAAAAAAAAdU/SYO259Bld24Lzhvsn7UfJu8XT7NGnpOBQCK8BGAsYHg/s0/2020-09-10.png"/>

3. Each row of the table represent different trials. The default metrics is the accuracy. Trial with ID 'DKqsP' is the best in this example 

4. The directory of the trained model is specified in the user training program. Here are the output directories of training programs:

   | Training program | output directory                           |
   | ---------------- | ------------------------------------------ |
   | ELMO             | {working_dir}/output_model                 |
   | mBart            | {working_dir}/output_model                 |
   | MASS             | {working_dir}/checkpoints/mass/pretraining |

5. With output directory and trial ID (DKqsP), you can check the trained model by going to directory 

```
{output directory}/DKqsP 
```

## Specify GPU 

The number of GPU and the index of GPU are speicified in `config.yml` in the working directory. You can change `gpuNum` and `gpuIndices` to specific the gpu that the training program runs on. For details, please check the comments on the `config.yml`

## Execute trials in parallel

To speed up the tuning process, you can execute the trials in parallel. 

For example, you have 8 GPUs. You can execute 8 trials in parallel. Each trial is allocated 1 GPU.

To do so (use ELMO's config as example)

1. Edit {elmo_working_dir}/config.yml

```yml
cat << EOF > config.yml
authorName: lscm
experimentName: elmo
# Specify the number of trials execute in parallel, here we execute 8 trials in parallel.
trialConcurrency: 8

# Specify the maximum runing time, we specify 1 week here
maxExecDuration: 40h 
maxTrialNum: 9999
trainingServicePlatform: local
searchSpacePath: search_space.json
useAnnotation: false
advisor:
  builtinAdvisorName: CUHKPrototypeTunerV2
  classArgs:
    num_epochs: 25
trial:
  command: python ./bin/train_elmo.py --train_prefix=./data/one_billion/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* --vocab_file ./data/vocab-2016-09-10.txt --save_dir ./output_model
  codeDir: .
  # specific the number of GPU used by each trial, here each trial use 1 GPU.
  gpuNum: 1
localConfig:
  # specific index of GPU used by the experiment, just like CUDA_VISIBLE_DEVICES
  # since we set 8 trials run in parallel and each trial need 1 GPU, total visible 
  # GPU device shoule be 8.
  gpuIndices: "0,1,2,3,4,5,6,7"
EOF
```



## Debugging

### NNI Error

Usually it shows up like this:

![Error Training service error: GPU not available. Please check your CUDA  configuration · Issue #2463 · microsoft/nni · GitHub](https://user-images.githubusercontent.com/23012102/82327761-723ddd80-9a11-11ea-9801-e42ba41b1321.png)

Then you can go to check the ``dispatcher.log`` and ``nnimanager.log`` to see if there is any report about this error.

To check ``dispatcher.log`` and ``nnimanager.log``, click ``Download`` -> ``Log files``: 

![Fix Python Neural Network Intelligence (NNI) Trial Jobs Status is Failed -  Python NNI Tutorial](https://lh3.googleusercontent.com/-NNAW85KsnLs/X6U0zru4bKI/AAAAAAAAAk0/2FGmNf1__TkvVNg0lORbd0Tb3z2WohCPgCK8BGAsYHg/s0/2020-11-06.png)

Then you can see the dispatcher.log and nnimanager.log:

![problem](https://user-images.githubusercontent.com/39946575/78686785-09e5e180-7926-11ea-9282-07a338b7a589.png)

### Trial Error

Usually it shows up like this:

![Fix Python Neural Network Intelligence (NNI) Trial Jobs Status is Failed -  Python NNI Tutorial](https://lh3.googleusercontent.com/-17ORUYET3Sc/X6UzVllmWfI/AAAAAAAAAko/ntLMr1vYmW4MatxbFx6K7XMmS0gW3zFNwCK8BGAsYHg/s0/2020-11-06.png)

You can check the trial output directory to track the error.

Click the failed trail -> Log:

![](https://lh3.googleusercontent.com/-atXbfdTeb-M/X4bAFeI1DkI/AAAAAAAAAh0/Cfrh7mZWXKU2kXAXJ2XTg6AgGOq12VpWgCK8BGAsYHg/s0/2020-10-14.png)

Clicking the two blue buttons will show you the corresponding contents. Or once you want to access those trial files, you can enter the ``Log path``:

![All trails are getting failed · Issue #1367 · microsoft/nni · GitHub](https://user-images.githubusercontent.com/8463288/62023228-82769900-b202-11e9-98b4-0a2dd02c8e8b.png)

``trial.log`` contain the training program standard output, such as user defined "print".

``stderr`` contain the error message generated from the training program.

## Upgrade from V1

1. Config file

   There is a necessary modification in config file. Please replace the corresponding codes according to the following example.

   ### Original (CUHKPrototypeTuner):

   ```yml
   tuner:
     builtinTunerName: CUHKPrototypeTuner
   ```

   ### New (CUHKPrototypeTunerV2):

   ```yml
   advisor:
     builtinAdvisorName: CUHKPrototypeTunerV2
     classArgs:
       num_epochs: 40
   ```

   ``num_epochs`` refers to the maximum or commonly-used number of epochs in your training, e.g. 40.

2. Directory of model checkpoint:

   The directories storing model checkpoints are renamed to ``{trialID}-{trialEpochs}-{configID}``. You can track the trail with its unique trial ID or specific budget.

   ```bash
   # tuner: CUHKPrototypeTuner
   # format: {output directory}/trialID
   # example:
   {output directory}/DKqsP
   
   # tuner: CUHKPrototypeTunerV2
   # format: {output directory}/{trialID}-{trialEpochs}-{configID}
   # example:
   {output directory}/cyXHO-3-5dd5dbcbbc1983b7
   ```

   During tuning, the tuner would maintain and may utilize such directories. Please do not delete them unless the training is finished or you are sure about such deletion.

