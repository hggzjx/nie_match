# mie_match
![image](https://github.com/hggzjx/nie_match/assets/103107137/69e18fff-3d03-474f-b3c3-96c9de389c6d)<br>

Model
---
<img width="3037" alt="Figure2" src="https://user-images.githubusercontent.com/103107137/228115682-9ae8c4f4-fe9e-4b90-be57-5d9f9be02c81.png"><br>

Overview of MIE-Match. (a) Model Structure; (b) Encoder; (c) Self-Fusion Module (SFM), comprising two options; (d) Interactive-Fusion Module (IFM), comprising two options.
How to run the model in the project
---
#tensorflow >= 1.14<br>
#keras >= 2.3.1<br>
python train.py<br>

Training with your own data
---
Process your own dataset into a csv file with three fields for header sentence1, sentence2, label, as shown below:<br>
<img width="136" alt="image" src="https://user-images.githubusercontent.com/103107137/228113859-4f78b407-718e-46c7-9308-4e47c6a385dc.png"><br>
Split the data into train.csv, dev.csv, test.csv and put them in . /datasets/dataset folder.<br>
Here are some commonly used text matching and natural language inference datasets:<br>
1.SNLI (Stanford Natural Language Inference): This dataset is a widely-used natural language inference dataset provided by Stanford University. It contains around 50,000 labeled pairs of text, including three labels: entailment, contradiction, and neutral.<br>
Download Link: https://nlp.stanford.edu/projects/snli/<br>
2.QQP (Quora Question Pairs): This dataset is created by Quora and consists of pairs of questions from the Quora website. The task is to determine if two questions are semantically similar or not.<br>
Download Link: https://www.kaggle.com/c/quora-question-pairs/data<br>
3.MRPC (Microsoft Research Paraphrase Corpus): This dataset is created by Microsoft Research and consists of sentence pairs that are labeled as either paraphrases or non-paraphrases. The sentences were extracted from news articles, and the dataset is commonly used for evaluating text matching and paraphrasing models.<br>
Download Link: https://www.microsoft.com/en-us/download/details.aspx?id=52398<br>
4.STS (Semantic Textual Similarity) Dataset: This dataset is a collection of sentence pairs that are labeled with a similarity score indicating the degree of semantic similarity between the two sentences. The dataset includes both news articles and tweets, and it is often used for evaluating text matching and semantic similarity models.<br>
Download Link: https://ixa2.si.ehu.es/stswiki/index.php/Main_Page<br>
5.SICK (Sentences Involving Compositional Knowledge) Dataset: This dataset is a collection of sentence pairs that are labeled as either entailment, contradiction, or neutral. The sentences were designed to test compositional knowledge, which refers to the ability to combine words and phrases to create new meanings. The dataset includes both simple and complex sentence structures, and it is commonly used for evaluating natural language inference models.<br>
Download Link: http://clic.cimec.unitn.it/composes/sick.html<br>
6.MNLI (Multi-Genre Natural Language Inference): This dataset is also provided by Stanford University and includes over 400,000 labeled pairs of text. It includes a wider range of text genres than SNLI and also has three labels: entailment, contradiction, and neutral.<br>
Download Link: https://cims.nyu.edu/~sbowman/multinli/<br>
7.GLUE (General Language Understanding Evaluation): This is a collection of nine natural language understanding tasks that can be used to evaluate language models. The tasks include sentence classification, text matching, and paraphrasing.<br>
Download Link: https://gluebenchmark.com/<br>
8.RTE (Recognizing Textual Entailment): This dataset is a collection of pairs of text created for the purpose of evaluating natural language inference systems. The task is to determine if a piece of text entails another piece of text.<br>
Download Link: https://aclweb.org/aclwiki/Recognizing_Textual_Entailment_(RTE)_Datasets<br>

Hyperparameter Setting
---
Set the parameters of the corresponding model in the corresponding position in the train.py script.The Implementation Details are shown as below：<br>
The experiments are conducted using the BiLSTM and BERT as the encoders of the MIE-Match model framework respectively. The MIE-Match model has different hyperparameter settings on different datasets, and some common hyperparameters are listed here as follows. The parameters of the pre-trained BERT are Layer = 24, Hidden = 1024, Head = 16, the size of the BiLSTM encoder is 64, the size of the BiLSTM used in the model subject is 128, and the size of the hidden layer of the fully connected layer MLP in the main body of the MIE-Match is 64. Cross-entropy is used as the loss function in the experiments, using $\beta_1 = 0.9$ and The Adam optimizer with $\beta_2 = 0.999$ is trained for 30 epochs, and the optimal model is selected using the early stop mechanism. 
