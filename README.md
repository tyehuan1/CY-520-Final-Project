# CY-520-Final-Project
Identify malware variants from API call sequences using ML

## Repository Structure
NOTE: Some subfolders in the directory are not listed (e.g., src.results.original, models.xgboost_olivera). These correspond to old models / data that were created when testing to try to improve generalizability of findings and are simply kept in the repository for reference and posterity.
```
CY-520-Final-Project/
├── config.py                   # Central configuration (paths, hyperparameters, random seed)
├── data/                       # Raw datasets (Mal-API-2019, MalBehavD-V1, WinMET, Olivera)
├── cache/                      # Cached intermediate results (TF-IDF matrices, vocabularies)
├── models/                     # Trained model artifacts
│   ├── xgboost_v2/             # XGBoost classifier
│   ├── lstm_v2/                # LSTM classifier
│   └── ensemble_v2/            # Confidence-gated ensemble
├── src/
│   ├── data_labeling/          # VirusTotal API labeling pipelines
│   │   ├── virustotal_labeler.py       # Primary dataset (Mal-API) VT labeling
│   │   └── olivera_vt_labeler.py       # Olivera dataset VT labeling
│   ├── data_loading/           # Dataset loading and preprocessing
│   │   ├── data_loader.py              # Primary/secondary dataset loaders
│   │   ├── preprocessing.py            # API call sequence cleaning and tokenization
│   │   ├── api_categories.py           # API call category mappings
│   │   ├── extract_winmet.py           # WinMET dataset extraction (full JSON data extremely large)
│   │   ├── build_olivera_dataset.py    # Olivera dataset construction (100 calls, no consecutive duplicates)
│   │   └── build_no_trojan_dataset.py  # Trojan-excluded datasets variant builder
│   ├── model_training/         # Model architectures and training scripts
│   │   ├── feature_engineering.py      # TF-IDF + statistical + category + bigram features
│   │   ├── xgboost_model.py           # XGBoost model wrapper
│   │   ├── lstm_model.py              # LSTM model wrapper
│   │   ├── ensemble_model.py          # Confidence-gated XGBoost/LSTM ensemble
│   │   ├── train_xgboost_v2.py        # V2 XGBoost training script
│   │   ├── train_lstm_v2.py           # V2 LSTM training script
│   │   └── train_ensemble_v2.py       # V2 ensemble training script
│   ├── evaluation/             # Evaluation and experiment scripts
│   │   ├── metrics.py                 # Metrics computation and plotting utilities
│   │   ├── evaluate_v2.py             # Full V2 evaluation suite (all datasets)
│   │   ├── evaluate_generalizability.py # Cross-dataset generalizability testing
│   │   └── evaluate_models.py         # Original model evaluation
│   ├── deployment/             # Gradio web application
│   │   └── app.py                     # Interactive classifier with SHAP explanations
│   └── utils.py                # Shared logging utility
├── tests/                      # Unit tests (mirrors src/ layout)
├── results/                    # Evaluation outputs (metrics JSON, plots, confusion matrices)
│   ├── v2/                     # V2 model results
│   │   ├── MalAPI/             # Primary dataset evaluation
│   │   ├── MalBehavD/          # MalBehavD generalizability results
│   │   └── WinMET/             # WinMET generalizability results
│   └── olivera/                # Olivera dataset results
├── deployment/                 # Docker deployment files
│   ├── Dockerfile
│   └── requirements.txt
```

## Problem Identification
As the time from initial access to exploitation continues to decrease, being able to identify malware as quickly as possible has become increasingly vital. However, in recent years malicious actors have increasingly disguised their malwares techniques, using legitimate tools as much as possible to avoid detection. This subterfuge has necessitated a stronger focus on real-time analysis, identifying adversaries through the actions they take, rather than just known signatures. This approach has the added benefit of being able to detect zero-days and other novel attacks. Behavioral analysis such as this can take several forms, using heuristics or sandboxing to break down what an application is trying to do. However, one way of detecting malicious actions is through an application’s API calls. 

When an application (benign or malware) interacts with the Windows OS, it calls Windows API functions. These API calls are how software requests services from the OS, e.g., opening files, allocating memory, connecting to the network. These API calls are a normal function of any software, and most API calls are inherently ambiguous. CryptEncrypt could be used as part of a ransomware operation, but it could also be a user request to protect a confidential file. However, the total sequence of API calls an application makes can reveal a lot about intent. 

While my model is necessarily retrospective in nature, this kind of analysis can also be used to classify an application’s behavior in real-time. Having a model providing information on the type of malware the software behavior is associated with as API calls come in can give evolving information, stopping malware before it can complete its goal. Advanced EDR tools do behavioral analysis like this, operating on live endpoints to provide evolving information as an application takes actions. My model wouldn’t sit on endpoints, instead it operates in tandem with a sandbox. When a suspicious file is flagged, for example by an email gateway or EDR, it can then isolate it and run it in a sandbox. This model would then be downstream of the sandbox where it’s fed the API calls, but upstream of the SIEM/SOAR which it then provides its analysis and enrichment to. Classification errors for malware families are a particular cocnern as they can waste time or even lead to incorrect solutions being implemented. For example, if a backdoor is labeled as a trojan the attacker may still be able to regain access even after the response concludes.

This model helps protect against all types of threat actors using malware, from script kiddies to APTs. The primary use case is for enterprise environments using Windows endpoints, such as computers and servers using the Windows OS. However, it is not necessarily infallible. Sometimes the model will have low confidence, requiring an analyst to manually review. This becomes even more likely if an actor knows this behavioral analysis is occurring and takes steps to evade it. Attackers can enter benign API calls in between the malicious ones reducing the signal and breaking up bigram and trigram patterns. They could also use direct system calls, bypassing what my model could actually see. Another evasion technique would have the malware try to detect if it’s in a sandbox and then run benign processes if it knows it’s being observed

## Data
Primary Dataset: Mal-API-2019 
Title: Data augmentation based malware detection using convolutional neural networks
Authors: Catak, Ferhat Ozgur and Ahmed, Javed and Sahinbas, Kevser and Khand, Zahid Hussain (2019)
Downloaded From: https://github.com/ocatak/malware_api_class
Notes: API call sequences were extracted by running malware in Cuckoo Sandbox and recording behavior. No preprocessing was mentioned by the authors. Labels came from some aggregation / weighting of VirusTotal vendor labels (full process not described by authors)

Secondary Dataset: MalBehavD-V1
Title: API-MalDetect: Automated malware detection framework for windows based on API calls and deep learning techniques
Authors: Maniriho, Pascal and Mahmood, Abdun Naser and Chowdhury, Mohammad Jabed Morshed (2023)
Downloaded From: https://github.com/mpasco/MalbehavD-V1
Notes: Similarly extracted from Cuckoo Sandbox, some sort of processing was likely done as median sequence length was 33 (way lower than the other datasets). Labels collected from VirusTotal (Virus and Worm barely represented)

Secondary Dataset: WinMET
Title: WinMET Dataset
Authors: Raducu, R., Villagrasa-Labrador, A., Rodríguez, R. J., & Álvarez, P (2025)
Downloaded From: https://zenodo.org/records/16414116
Notes: "WinMET dataset contains the execution traces generated with CAPE sandbox after analyzing several malware samples. The execution traces are valid JSON files that contain the spawned processes, the sequence of WinAPI and system calls invoked by each process, their parameters, their return values, and OS accesed resources, amongst many others." Notably, this also contains information on the malware family (e.g., Emotet, Redline) that was then classifed based on its family as one of the Mal-API-2019 labels

Secondary Dataset: Olivera
Title: Behavioral Malware Detection Using Deep Graph Convolutional Neural Networks. TechRxiv. Preprint.
Authors: Oliveira, Angelo; Sassi, Renato José (2019)
Downloaded From: https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences
Notes: Preprocessed so that "Each API call sequence is composed of the first 100 non-repeated consecutive API calls associated with the parent process, extracted from the 'calls' elements of Cuckoo Sandbox reports." Labels collected from VirusTotal 


## Generalizability
Problems Encountered:
API call format: Mal-API-2019 records lowercase function names (createfilew), MalBehavD uses mixed-case (CreateFileW), and WinMET uses its own instrumentation format extracted from CAPE.

Sequence length: Mal-API sequences average ~200-300 calls; MalBehavD had a median of 33 calls and WinMET sequences varied significantly but had a much higher median than Mal-Api.

Class distribution: The primary dataset has 8 classes. WinMET lacks Adware and Dropper; MalBehavD has very few Adware (29) and Downloader (18) samples; Olivera has less than 20 samples of both Virus and Worms (and is oddly heavy on Adware). Only overlapping classes were evaluated.

Sandbox environment: As described above Mal-Api, MalBehavD, and Olivera samples were extraced from Cuckoo Sandbox, WinMET was extracted from CAPE

Missing Labels: None of the 3 secondary datasets had matching class labels to Mal-Api. 

Preprocessing Compatibility Steps:

Unified API call normalization (lowercase, stripped suffixes like A/W/Ex, and manually matched some WinMET calls to their counterpart from Mal-Api)

Consistent tokenization and TF-IDF vocabulary (vocabulary built from training data, applied to secondary datasets — unseen tokens mapped to zero)
    One test run explored removing the TF-IDF vocabulary from the XGBoost training entirely (did not improve results)

Identical feature engineering pipeline (TF-IDF + 9 statistical features + 8 category features + 64 bigram features = 5,081 total features)

More than 5 duplicates of the same call in a row were removed to capture repeated call context while limtiing samples that had 1000's of the same call repeated.

LSTM sequences padded/truncated to same fixed length (500) to deal with sequence length imbalance
    For one generalizability test involving the Olivera dataset, one variant tried limiting the training data to that same preprocessing of 100 calls only and consecutive duplicates removed.

Class labels were collected for Olivera and MalBehavD using the VirusTotal script to query its API and string searching for the different classes. Steps were taken to avoid overcounting Trojan which is often a generic entry from vendors or is combined with other more distinctive behaviors such as Trojan.Backdoor. A few malware families (e.g., padodor) were manually classified. Because the WinMET dataset only contained those malware families, the top 25 families were extracted from the data (with a maximum of 300 samples each) and then they were manually classified into one of the Mal-Api labels.


## Models Chosen

XGBoost:

Vocab was constructed by assigning each unqiue call in order an integer plus an UNK and PAD token. Then API calls were replaced with their encoded value.
Feature engineering was done including a TF-IDF vectorizer (top 5000 unigrams), statistical features (e.g., unique ratio, shannon frequency), API category features, and category-to-category transition features (8x8 matirx showing, for example, network to file transition). Normalizing was done so that sequence length imapct was minimal (which was important given the secondary datasets and the primary dataset had wildly varying lengths). Ended up with a 5,081 feature vector based on the top 5000 unigrams and the other features described. Hyperparamters were selected using RandomizedSearchCV
XGBoost was ideal because most malware only uses a small subset of the many tracked API calls (only ~280 unique calls in the Mal-Api samples), so the TF-IDF matrix is very sparse which a tree model should handle well. This matched from a paper analyzing this data which found the best performance was at 69% for XGBoost. XGBoost is also good with class imbalances like can be common with malware samples and it allows one to view the SHAP results so you can understand exactly which features are driving a classification (for example, the bigram of cryptencrypt and writefile could be a sign of a ransomware attack that the SHAP could show). Further, XGboost is good more generally with identifying the combination of features throughout the sequence and learning when that is suspicious across this limited vocabulary.
Potential pitfall is that XGBoost does not understand order vs a model like LSTM that does. A sequence like connect → recv → writefile looks identical to writefile → recv → connect to XGBoost.


LSTM:

Vocab was constructed by assigning each unqiue call in order an integer plus an UNK and PAD token. Then API calls were replaced with their encoded value. The integer sequence was then truncated/padded to 500 tokens and for long sequences a sliding window was used on overlapping windows of 500 token windows to pick up more of the context (important for WinMET which had much longer sequences).
Tuned to have embedding layer (128) and two bidirectional LSTM layers (128 and 64) allowing the sequnece do be processed in both directions learning sequential dependencies and other temporal patterns. Spatial dropout was used after embedding and recurrent dropout in the LSTM layers to prevent overfitting while preserving memory.
LSTM was chosen to complement XGBoost well because it processes the raw API call sequence as an ordered time series allowing it to understand temporal patterns (like that Downloaders call network APIs before file-write APIs). These ordering patterns are invisible to XGBoost. It also understand variable length contexts by accumulating information across the sequence in both directions and is an RNN instead of a tree-based algorithm providing a different methodology.
Potential pitfalls are that LSTMs are harder to interpret as I didn't have something like SHAP, they require padding or truncating to match the fixed input length, and they generally performed worse on the original dataset (0.57 vs 0.70 F1) suggesting that at least for Mal-Api what APIs are called matters more than when they're called. One other thing of note is that Mal-Api only contained around 280 unqiue API calls, an extremely small vocabulary for the language processing of the LSTM model.

Ensemble: The ensemble model was chosen in the hopes that XGBoost and LSTM would make errors in different ways, and that when they did so the models confidence would be lower. Thus by combining the two performance could be imrproved by having the benefit of different types of features including sequential information and TF-IDF vocab and combining temporal information with frequency based classifications. However, the models were too correlated for this to have much success in most cases except for with the Olivera v2.
Training was done by combining the two models with a probability gated threshold that leaned towards XGBoost because of its better performance. Final threshold chosen ended up being 50% where if XGBoost had 50% confidence it would use that alone, but if it was below it would blend the two based on the class F1 scores.

Optimizations:
Many different optimization strategies were attempted to maximize model performance, especially after the low generalizability was discovered. A few (but not exhaustive list) examples of this include experimenting with the different hyperparameter optons I would allow RandomizedSearchCV to test (XGBoost), testing the performance on different sequence legnths (LSTM), exploring whether trigrams should be included (XGBoost), testing different statistical features (e.g., sequence length, top 5 features ratio of total), removing trojan from either the generalizability dataset or from both that and the training dataset, and exploring sliding windows to capture more context.

## Analysis

Performance Metrics Used:

Macro F1-score (primary metric): Averages F1 across all classes equally, ensuring minority class performance matters. Critical because some of the malware generalizability datasets are heavily imbalanced.
Per-class F1, Precision, Recall: Reveals which specific families the model can/cannot generalize.
Accuracy: Reported but de-emphasized due to class imbalance.
ROC AUC (weighted): Measures discriminative ability across probability thresholds.


Results - Primary Dataset (Mal-Api):

Model	    Accuracy	Macro F1
XGBoost	    0.689	    0.702
LSTM	    0.557	    0.570
Ensemble	0.681    	0.692

Results - Secondary Datasets:

Model	 MalBehavD F1  WinMET F1  Olivera F1  MalBehavD Drop  WinMET Drop  Olivera Drop
XGBoost	 0.111	       0.092      0.134       -84.1%	      -86.9%       -81.0%
LSTM	 0.115	       0.092      0.107       -79.9%	      -84.0%       -81.2%
Ensemble 0.091	       0.083	  0.172       -86.9%	      -88.1%       -75.1%

Key Findings:
All models suffer 75-88% F1 drops on secondary datasets, implying that API-call-based classifiers are heavily data-dependent. The models seem to learn the frequency distribution of API calls as seen for one specific time and dataset, not truly abstract behavioral patterns. While my results were actually slightly better for the primary dataset on XGBoost (70% vs the paper's 69%) the Mal-Api authors (nor any of the other papers examined) did not due any generalizability testing. it is possible this is just a limitation of the data (different sandboxes, timeframes, sequence lengths) or of the specific labels used (Trojan doesn't describe a behavior it describes a delivery mechanism and is often a generic label on VirusTotal, most malware samples contain multiple capabilities so a spyware could also install a backdoor) but it still raises questions about the original papers findings on this subject.

LSTM generalizes slightly better than XGBoost (80-84% drop vs 81-87%), likely because temporal ordering patterns (e.g., Downloaders always calling network APIs before file-write APIs) transfer across sandboxes better than the TF-IDF distributions.

Virus and trojan overclassification dominates all three secondary datasets. This happens for slightly different reasons:
Trojan cosine similarity was the most similar on average to all 7 other classes (which makes sense given the label discussion of delivery vs behavior). This mean samples from many classes got sent there because their features weakly match Trojan's average profile.
Virus cosine similarity was the least similar on average to all 7 other classes. This means samples get sent there because their features don't match anything the model learned, and virus maps most closely to the unknown.

Bright Spot 1: Downloader and Spyware contain more distinctive behavior. Even across sandboxes when testing the generalizability of the WinMET dataset the LSTM model achieved 0.442 F1 on Downloaders, suggesting something about the sequential behavioral patterns (maybe network→disk write ordering) does transfer across environments. Similarly the Olivera generalizability test achieved 0.639 F1 on Spyware (XGBoost).

Bright Spot 2: Training performance was very strong with the model getting a 70% macro F1 using XGBoost (across the different optimziations this varied from around 62% to 70%) which matched with the findings of the paper analyzing the Mal-Api data from which it was first downloaded. However, it was not learning generalizability patterns.


## Production Deployment
Deployment Environment: Hugging Face Spaces (Gradio)

Interface Features:

The Gradio web application (src/deployment/app.py) provides:

Two input methods. Either paste API call sequences as text or upload a text file
Three-model inference: Runs XGBoost, LSTM, and Ensemble simultaneously, and shows each models predicted family and confidence.
Confidence bar chart: Grouped bar chart comparing all three models confidence distributions across the different families
SHAP explanations: Waterfall chart showing the top 15 features driving the XGBoost prediction (which features pushed toward/against the predicted class)
Review flag: When all three models have confidence below 0.40, the interface flags the sample for manual analyst review.

Packaging:

The model is packaged using containerized deployment on Hugging Face Spaces (Gradio SDK), which auto-builds a Docker container from the requirements.txt. A manual Dockerfile is also provided in deployment/Dockerfile for local Docker deployment with all dependencies, model artifacts, and cached vocabularies
Requirements: deployment/requirements.txt requires versions for Gradio, NumPy, scikit-learn, XGBoost, TensorFlow, SHAP, Matplotlib, Seaborn

Deployment Steps:

Build the Docker image: docker build -f deployment/Dockerfile -t malware-classifier .
Run the container: docker run -p 7860:7860 malware-classifier
Access the interface at http://localhost:7860
For Hugging Face Spaces: create a Gradio SDK Space, push the code/models/requirements.txt, and it auto-builds a container"

Real-Time Inference:

Models load once at startup and are cached in memory
Single-sample inference completes in 5-8 seconds
