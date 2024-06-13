## Evaluate Foundational Model Quality  
  
### Motivation
  
Public benchmarks are often used to assess foundation model performance across a wide variety of tasks. However, the fine print of many of these test reveals inconsistent prompting methodology which leads to confusing and unreliable results. The goal of this repository is to provide a transparent, flexible, and standadized method to repeatably compare different foundaiton models and model versions. This repsoitory is designed to be executed to uniquely compare _my_ existing model to _my_ challenger model as opposed to relying on public benchmarks executed on a model instantiation managed by some other entity.
  
### Benchmark Datasets  
  
**MMLU**
  
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability.  
  
Data size = 14,402 rows
  
We have grouped some of the taks into borader categories for easier targeted execution. These categories are: STEM, Medical, Business, Social Sciences, Humanities, and Other.
  
[Paper](https://arxiv.org/pdf/2009.03300) | [HuggingFace Dataset](https://huggingface.co/datasets/cais/mmlu)
  
  
**Truthful QA**
  
TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts.  
    
Data size = 814 rows
  
[Paper](https://arxiv.org/pdf/2109.07958) | [HuggingFace Dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa) | [GitHub](https://github.com/sylinrl/TruthfulQA)
    
**PubMedQA**
  
The task of PubMedQA is to answer research questions with yes/no/maybe _(e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?)_ using the corresponding abstracts. PubMedQA has 1k expert labeled instances.  
 
Data size = 1,000 rows
  
[Paper](https://arxiv.org/pdf/1909.06146) | [HuggingFace Dataset](https://huggingface.co/datasets/qiaojin/PubMedQA) | [Website](https://pubmedqa.github.io/) | [GitHub](https://github.com/pubmedqa/pubmedqa)
