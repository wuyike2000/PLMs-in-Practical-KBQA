# An Empirical Study of Pre-trained Language Models in Simple Knowledge Graph Question Answering

> **Abstract**
Large-scale pre-trained language models (PLMs) such as BERT have recently achieved great success and become a milestone in natural language processing (NLP). It is now the consensus of the NLP community to adopt PLMs as the backbone for downstream tasks. In recent works on knowledge graph question answering (KGQA), BERT or its variants have become necessary in their KGQA models. However, there is still a lack of comprehensive research and comparison of the performance of different PLMs in KGQA. To this end, we summarize two basic KGQA frameworks based on PLMs without additional neural network modules to compare the performance of nine PLMs in terms of accuracy and efficiency. In addition, we present three benchmarks for larger-scale KGs based on the popular SimpleQuestions benchmark to investigate the scalability of PLMs. We carefully analyze the results of all PLMs-based KGQA basic frameworks on these benchmarks and find that knowledge distillation techniques and knowledge enhancement methods in PLMs are promising for KGQA. We have released the code and benchmarks to promote the use of PLMs on KGQA.

This is the accompanying code & benchmarks for the paper "[An Empirical Study of Pre-trained Language Models in Simple Knowledge Graph Question Answering](null)"

## Package Description
Important files/folders are described as follows:

```
PLMs-in-Practical-KBQA/main/
├─ entity_detection/: Entity Detection
    ├─ ner.py: Training and inferencing PLMS-based NER, except GPT2
    ├─ train.py: Training and inferencing GPT2-based NER
    ├─ner_label.py: Generating labels for similarity-based relation prediction
├─ entity_linking/: Entity Disambiguation based on Vanilla method
    ├─ entity_linking.py: Disambiguat entity using vanilla method. 
├─ entity_disamb/: PLMs-based Entity Disambiguation. Before running the PLMS-based ED model, you need to run entity_linking.py
    ├─ candidate_convert.py: Step 1. Preprocessing the output of entity_linking.py
    ├─ run_disamb.py: Step 2. Training and inferencing PLMS-based ED. 
    ├─ result_convert.py: Step3. Evaluating PLMS-based ED results.
├─ rel_prediction/: Classfication-based relation prediction
    ├─ classify.py: Training
    ├─ test_re.py: Testing
├─ pathranking/: Pre-processing of similarity-based relation prediction
    ├─ testdata.py: Constructing dictionary and replacing subject mentions
    ├─ test_json.py: Generating JSON files for testing
├─ rel_similarity/: Similarity-based relation prediction
    ├─ train.py: Training and inferencing similarity-based relation prediction
├─ evidence_integration/: Answer Query
    ├─ final.py: Evaluating the final accuracy
├─ pretrain: PLMs caches.
├─ indexes: All dictionaries
├─ mydata: Dataset
├─ mydata1: Dataset
├─ kb_105M: Benchmark
├─ kb_202M: Benchmark
```

Note that you need to modify following parametes to select PLMs and Benchmarks: SCALE->[small, medium1, medium2, large], TYPE-> the PLM names, MODEL-> the PLM cache names in FOLDER PLMs-in-Practical-KBQA/main/pretrain

Downloading related caches from https://drive.google.com/drive/folders/1dBHtni2cA4ASMdEGQbn0n5cB-ajc9aXX?usp=sharing

Downloading PLMs caches: [BERT](https://huggingface.co/bert-base-uncased), [RoBERTa](https://huggingface.co/roberta-base), [XLNET](https://huggingface.co/xlnet-base-cased), [GPT2](https://huggingface.co/gpt2), [ALBERT](https://huggingface.co/albert-base-v2), [DistilBERT](https://huggingface.co/distilbert-base-uncased), [DistilRoBERTa](https://huggingface.co/distilroberta-base), [LUKE](https://huggingface.co/studio-ousia/luke-base), [KEPLER](https://github.com/THU-KEG/KEPLER)



## Contact
Please consider creating a new issue. We will respond to your questions within a few days.