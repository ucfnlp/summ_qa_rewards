# Guiding Extractive Summarization with Question-Answering Rewards

This repository contains ready to run code for extractive summarization following our paper from NAACL 2019.  We ask that you please cite our paper if you make use of our findings or code.
```
@inproceedings{PLACEHOLDER,
  title={Guiding Extractive Summarization with Question-Answering Rewards},
  author={Arumae, Kristjan and Liu, Fei},
  booktitle={Proceedings of NAACL 2019},
  year={2019}
}
```
The code and instructions allow for the following:
1. Pre-processing the CNN/Daily Mail dataset for our models.  This includes steps for question generation using NER, SUBJ/OBJ, and ROOT words.  Please refer to our paper for details.
2. Training a full model from scratch.
3. Sample output from models as reported in our publication.

## Code Execution
#### Requirements 
* [Theano version 1.0.1](http://deeplearning.net/software/theano/install.html)
* [Stanford CoreNLP Toolkit](https://stanfordnlp.github.io/CoreNLP/) (Pre-Processing only)
  * [SR Parser jar](https://nlp.stanford.edu/software/stanford-srparser-2014-10-23-models.jar) (download jar into Stanford Core NLP root directory) 
* [pyrouge](https://pypi.org/project/pyrouge/)
* Cuda 9.0

#### Data Pre-Processing
We process CNN and Daily Mail separately.  The following details the data processing pipeline for CNN.
1. Map and pre-process data for Stanford CoreNLP input. This separates highlights and articles. 
    ```bash
    python constituency_parse.py 
           --parsed_output_loc "<path to output>" \
           --process True \
           --source cnn \
           --raw_data "<path to input data>"
    ```
    Navigate to the directory specified for output.  Highlights are split into several files, for memory overhead. For CNN there should be 11 files. 
    ```bash
    max=11
    for i in `seq 1 $max`
    do
        java -Xmx10g \
             -cp "<path to Stanford CoreNLP>/*" edu.stanford.nlp.pipeline.StanfordCoreNLP \
             -annotators tokenize,ssplit,pos,lemma,ner,parse \ 
             -parse.model "edu/stanford/nlp/models/srparser/englishSR.ser.gz" \
             -parse.nthreads 10 \
             -ner.nthreads 10 \
             -tokenize.whitespace true \
             -file highlights"$i".txt \
             -outputFormat "json"
    done
    ```

#### Training

```bash
PYTHONPATH=<path to repo> \ 
THEANO_FLAGS=device=cuda0,floatX=float32 \ 
python main.py \
               --load_model_pretrain True \  
               --pretrain False \
               --full_test True \  
               --load_model \  
               --source cnn \
               --max_epochs 25 \
               --coeff_cost_scale 1.0 \
               --coeff_adequacy 8 \
               --coeff_z 50 \
               --nclasses 15342 \
               --num_files_train 77 \  
               --num_files_dev 5 \
               --num_files_test 5 \ 
               --batch_dir ../data/batches_dm_400_ch_ner_cutoff_5/ \  
               --inp_len 400 \
               --generator_encoding lstm \  
               --word_level_c False \
               --batch 128 \
               --use_generator_h True \  
               --online_batch_size 20 \
               --rl_no_qa False \
               --z_perc 0.15 \
               --is_root False \ 
               --n 10
```

#### Example Output