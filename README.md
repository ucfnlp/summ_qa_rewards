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

To the best of our abilities we have cleaned the code to remove superfluous functionality.  Any questions/concerns can be directed towards [Kristjan Arumae](http://kristjanarumae.com/). (email: kristjan \<dot> arumae \<at> gmail \<dot> com)

## Code Execution
#### Requirements 
* [Theano v1.0.1](http://deeplearning.net/software/theano/install.html)
* [Stanford CoreNLP Toolkit v3.7.0](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip) (Pre-Processing only)
  * [SR Parser jar](https://nlp.stanford.edu/software/stanford-srparser-2014-10-23-models.jar) (download jar into Stanford Core NLP root directory)
* [CNN/Daily Mail tokenized input](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail)
* [GloVe Embeddings](http://nlp.stanford.edu/data/glove.6B.zip) 
* [pyrouge](https://pypi.org/project/pyrouge/)
* Cuda v9.0

#### Data Pre-Processing
We process CNN and Daily Mail separately.  The following details the data processing pipeline for CNN.
1. Map and pre-process data for Stanford CoreNLP input. This separates highlights and articles. 
    ```bash
    python constituency_parse.py 
           --parsed_output_loc <PATH_TO_OUTPUT> \
           --source cnn \
           --raw_data <path to input data>
    ```
    There now exists a filesystem with the root as your pre-specified \<path to output>.  There are also two files created which will serve as file lists for the next steps.   
    
2. We will now use Stanford CoreNLP on the highlights and articles separately.  From the highlights we need to acquire NER, ROOT, and SUBJ/OBJ (see § 3.3) to generate questions further down the pipeline.  From the articles we need constituency parse trees for input chunks (see § 3.1).
    
    *NOTE*: Stanford CoreNLP runs significantly faster, and with less memory, when the input is a file list.  This process is nonetheless slow. 
    ```bash
    java -Xmx10g \
         -cp "<path to Stanford CoreNLP>/*" edu.stanford.nlp.pipeline.StanfordCoreNLP \
         -annotators tokenize,ssplit,pos,lemma,ner,parse \ 
         -parse.model "edu/stanford/nlp/models/srparser/englishSR.ser.gz" \
         -parse.nthreads 10 \
         -ner.nthreads 10 \
         -tokenize.whitespace true \
         -filelist list_hl.txt \
         -outputFormat "json" \
         -outputDirectory <PATH_TO_OUTPUT>/highlights_scnlp/
    ```
    ```bash
    java -Xmx10g \
         -cp "<path to Stanford CoreNLP>/*" edu.stanford.nlp.pipeline.StanfordCoreNLP \
         -annotators tokenize,ssplit,pos,lemma,parse \ 
         -parse.model "edu/stanford/nlp/models/srparser/englishSR.ser.gz" \
         -parse.nthreads 10 \
         -tokenize.whitespace true \
         -filelist list_art.txt \
         -outputFormat "json" \
         -outputDirectory <PATH_TO_OUTPUT>/articles_scnlp/ \
         -ssplit.eolonly
    ```
   
3. The next processing step is the last general processing step.  This will:
    * Determine all entities present (SUBJ/OBJ, NER, and ROOT)
    * Determine text chunks. 
    * Map input tokens to vocabulary.
    * Create auxilliary output for testing.
    
    Depending on your goal with our models, after finishing this data processing step you will not need to repeat it and ones above. 

    ```bash
    python process_scnlp.py
 				--full_test True \
 				--raw_data_cnn <PATH_TO_OUTPUT> \
 				--vocab_size 150000 \
		        --chunk_threshold 5 \
 				--source cnn
    ```
4. The next step is most important for choosing whether to use the chunks previously create, as well as the QA type. (Bellow we use chunks, and NER)
    
    ```bash
    python low_level_process_data.py
				--full_test True \
				--vocab_size 150000 \
				--source cnn \
				--word_level_c False \
				--n 10 \
				--skip_root False \
				--use_root False \
				--use_obj_subj False \
				--use_ner True
    ```
5. The last data processing step is under batching.  This allows for separation of data processing and training completely.  The options we control at the batching level.
    ```bash
    python create_batches.py \
                    --embedding <PATH_TO_EMB>/glove.6B.100d.txt \
                    --vocab_size 150000 \
                    --word_level_c False \
                    --full_test True \
                    --batch 64 \
                    --online_batch_size 5 \
                    --source cnn \
                    --inp_len 400 \
                    --n 10 \
                    --sort sort
    ```
 
#### Training

```bash
PYTHONPATH=<PATH_TO_REPO> \ 
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