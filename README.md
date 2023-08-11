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
