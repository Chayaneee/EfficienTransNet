# EfficienTransNet
EfficienTransNet: An Automated Chest X-ray Report Generation Paradigm

# Introduction: An Automated Chest X-ray Report Generation Paradigm
This is the official repository of our proposed EfficienTransNet. The significance of chest X-ray imaging in diagnosing chest diseases is well-established in clinical and research domains. The automation of generating X-ray reports can address various challenges associated with manual diagnosis by speeding up the report generation system, becoming the perfect assistant for radiologists, and reducing their tedious workload. But, this automation's key challenge is to accurately capture the abnormal findings and produce a fluent as well as natural report. In this paper, we present a CNN-Transformers based automatic chest X-ray report generation approach named EfficienTransNet that prioritizes clinical accuracy and demonstrates improved text generation metrics. Our model incorporates clinical history or indications to enhance the report generation process and align with radiologists' workflow, which is mostly overlooked in recent research. On two publicly available X-ray report generation datasets, MIMIC-CXR, and IU X-ray, our model yields promising results on natural language evaluation and clinical accuracy metrics. Qualitative results, demonstrated with Grad-CAM, provide disease location information for radiologists' better understanding. Our proposed model emphasizes radiologists' workflow, enhancing the explainability, transparency, and trustworthiness of radiologists in the report generation process. 

# Proposed Pipeline
![Block_Diagram_EfficienTransNet](https://github.com/Chayaneee/EfficienTransNet/assets/54748679/66bce3f4-0ffe-468c-8124-cbe73ffd5f21)


# Data used for Experiments: 

We have used two publicly available datasets for this experiment.
  - [IU X-ray](https://openi.nlm.nih.gov/)
  - [MIMIC-CXR](https://physionet.org/content/mimiciii-demo/1.4/)

# Evaluation Metrics 
1. Word Overlap Metrics: BLEU-score, METEOR, ROUGE-L, CIDER
2. Clinical Efficiency (CE) Metrics: AUC, F1-score, Precision, Recall, Accuracy
3. Semantic Similarity-based Metrics: Skip-Thoughts, Average Embedding, Vector Extrema, Greedy Matching

# Quantative Results
Link to convert Latex to Markdown Table: https://tableconvert.com/latex-to-markdown

| \textbf{Datasets}               | \textbf{Models}                                      | \textbf{B1}         | \textbf{B4}         | \textbf{MTR}        | \textbf{R-L}        | \textbf{CDr}        | \textbf{FS}       | \textbf{PR}       | \textbf{RE}       | \textbf{ACC}      | \textbf{AUC}      |
|---------------------------------|------------------------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|                                 | Xu 2018~\cite{xue2018multimodal}                     | $0.464$             | $0.195$             | $\textbf{0.274}$    | $0.366$             | $-$                 | $-$               | $-$               | $-$               | $-$               | $-$               |
|                                 | Jing 2018~\cite{jing-etal-2018-automatic}            | $\textbf{0.517}$    | $\textbf{0.247}$    | $0.217$             | $\textbf{0.447}$    | $\underline{0.327}$ | $-$               | $-$               | $-$               | $-$               | $-$               |
|                                 | R2GEN 2020~\cite{chen2020generating}                 | $0.470$             | $0.165$             | $0.187$             | $0.371$             | $-$                 | $-$               | $-$               | $-$               | $-$               | $-$               |
|                                 | Omar 2021~\cite{alfarghaly2021automated}             | $0.387$             | $0.111$             | $0.164$             | $0.289$             | $0.257$             | $-$               | $-$               | $-$               | $-$               | $-$               |
|                                 | Nguyen 2021~\cite{nguyen2021automated}               | $\underline{0.515}$ | $\underline{0.235}$ | $\underline{0.219}$ | $\underline{0.436}$ | $-$                 | \underline{0.626} | \underline{0.604} | \textbf{0.649}    | \underline{0.937} | \textbf{0.877}    |
|                                 | R2GENCMN 2022~\cite{chen2022cross}                   | $0.475$             | $0.170$             | $0.191$             | $0.375$             | $-$                 | $-$               | $-$               | $-$               | $-$               | $-$               |
|                                 | DCL 2023~\cite{li2023dynamic}                        | $-$                 | $0.163$             | $0.193$             | $0.383$             | $0.586$             | $-$               | $-$               | $-$               | $-$               | $-$               |
|                                 | EfficienTransNet                                     | $0.488$             | $0.221$             | $0.206$             | $0.424$             | $\textbf{0.691}$    | \textbf{0.639}    | \textbf{0.647}    | \underline{0.631} | \textbf{0.942}    | \underline{0.859} |
| \multirow{9}{*}{\textbf{MIMIC}} | Liu 2019~\cite{liu2019clinically}                    | 0.313               | 0.103               | -                   | 0.306               | -                   | -                 | 0.309             | 0.134             | \underline{0.867} | -                 |
|                                 | Trans-Prog. 2021~\cite{nooralahzadeh2021progressive} | 0.378               | 0.107               | 0.145               | 0.272               | --                  | 0.308             | 0.240             | \textbf{0.428}    | -                 | -                 |
|                                 | Co-ATT 2021~\cite{liu-etal-2021-contrastive}         | 0.350               | 0.109               | 0.151               | 0.283               | -                   | 0.303             | 0.352             | 0.298             | -                 | -                 |
|                                 | Nguyen 2021~\cite{nguyen2021automated}               | \textbf{0.495}      | \textbf{0.224}      | \textbf{0.222}      | \textbf{0.390}      | -                   | \textbf{0.412}    | \underline{0.432} | \underline{0.418} | \textbf{0.887}    | \textbf{0.784}    |
|                                 | R2GENCMN 2022~\cite{chen2022cross}                   | 0.353               | 0.106               | 0.142               | 0.278               | -                   | 0.278             | 0.334             | 0.275             | -                 | -                 |
|                                 | CvT-DGPT2 2022~\cite{nicolson2022improving}          | 0.395               | 0.127               | 0.155               | 0.288               | \textbf{0.379}      | \underline{0.390} | 0.365             | \underline{0.418} | -                 | -                 |
|                                 | DCL 2023~\cite{li2023dynamic}                        | -                   | 0.109               | 0.150               | 0.284               | 0.281               | 0.373             | \textbf{0.471}    | 0.352             | -                 | -                 |
|                                 | EfficienTransNet                                     | \underline{0.468}   | \underline{0.195}   | \underline{0.199}   | \underline{0.363}   | \underline{0.359}   | 0.371             | 0.369             | 0.405             | 0.852             | \underline{0.676} |
