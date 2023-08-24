[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/humza909/Survery-Image-Mixing-and-Deleting-for-Data-Augmentation)

# A Comprehensive Overview of Large Language Models
This repo is for our paper: https://arxiv.org/abs/2307.06435

## Contents
- [Surveys](#surveys)
- [Pre-trained LLMs](#pre-trained-llms)
  - [General Purpose](#general-purpose)
  - [Coding](#coding)
  - [Scientific Knowledge](#scientific-knowledge)
  - [Dialog](#dialog)
  - [Finance](#finance)
- [Fine-tuned LLMs](#fine-tuned-llms)
  - [Instruction-tuned with Manually Created Datasets](#instruction-tuned-with-manually-created-datasets)
  - [Instruction-tuned with LLMs Generated Datasets](#instruction-tuned-with-llms-generated-datasets)

## Surveys
* Towards Reasoning in Large Language Models: A Survey, arXiv, 2022. [[Paper](https://arxiv.org/abs/2212.10403)]
* Emergent Abilities of Large Language Models, arXiv, 2022. [[Paper](https://arxiv.org/abs/2206.07682)]
* Several categories of Large Language Models (LLMs): A Short Survey arXiv, 2023. [[Paper](https://arxiv.org/abs/2307.10188)]
* Retrieving Multimodal Information for Augmented Generation: A Survey, arXiv, 2023. [[Paper](https://arxiv.org/abs/2303.10868)]
* Large Language Models in Medical Education: Opportunities, Challenges, and Future Directions, JMIR, 2023. [[Paper](https://mededu.jmir.org/2023/1/e48291/)]
* Language Model Behavior: A Comprehensive Survey, arXiv, 2023. [[Paper](https://arxiv.org/abs/2303.11504)]
* Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond, arXiv, 2023. [[Paper](https://arxiv.org/abs/2304.13712)]
* Beyond One-Model-Fits-All: A Survey of Domain Specialization for Large Language Models, arXiv, 2023. [[Paper](https://arxiv.org/abs/2305.18703)]
* A Survey on Large Language Models: Applications, Challenges, Limitations, and Practical Usage, TechRxiv, 2023. [[Paper](https://www.techrxiv.org/ndownloader/files/41501037)]
* Recent advances in natural language processing via large pre-trained language models: A survey, ACM Surveys, 2021. [[Paper](https://dl.acm.org/doi/abs/10.1145/3605943)]
* Complex QA and language models hybrid architectures, Survey, arXiv, 2023. [[Paper](https://arxiv.org/abs/2302.09051)]
* Challenges and Applications of Large Language Models, arXiv, 2023. [[Paper](https://arxiv.org/abs/2307.10169)]
* Augmented Language Models: a Survey, arXiv, 2023. [[Paper](https://arxiv.org/abs/2307.10169)]
* A Survey on Multimodal Large Language Models, arXiv, 2023. [[Paper](https://arxiv.org/abs/2306.13549)]
* A Survey on Evaluation of Large Language Models, arXiv, 2023. [[Paper](https://arxiv.org/abs/2307.03109)]
* A Survey of Large Language Models, arXiv, 2023. [[Paper](https://arxiv.org/abs/2303.18223)]
* ChatGPT for good? On opportunities and challenges of large language models for education, LID, 2023. [[Paper](https://www.sciencedirect.com/science/article/pii/S1041608023000195)]
* A Short Survey of Viewing Large Language Models in Legal Aspect, arXiv, 2023. [[Paper](https://arxiv.org/abs/2303.09136)]
* Aligning Large Language Models with Human: A Survey, arXiv, 2023. [[Paper](https://arxiv.org/abs/2307.12966)]
* A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT, arXiv, 2023. [[Paper](https://arxiv.org/abs/2302.09419)]
* Instruction Tuning for Large Language Models: A Survey, aeXiv, 2023. [[Paper](https://arxiv.org/pdf/2308.10792v1.pdf)]
## Pre-trained LLMs
### General Purpose
* **T5:** Exploring the limits of transfer learning with a unified text-to-text transformer, JMLR, 2020. [[Paper](https://arxiv.org/abs/1910.10683)]
* **GPT-3:** Language Models are Few-Shot Learners, NeurIPS, 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html?utm_medium=email&utm_source=transaction)]
* **mT5:** A Massively Multilingual Pre-trained Text-to-Text Transformer, NAACL, 2021. [[Paper](https://arxiv.org/abs/2010.11934)]
* **PanGu-alpha:** Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation, arXiv, 2021. [[Paper](https://arxiv.org/abs/2104.12369)]
* **CPM-2:** Large-scale cost-effective pre-trained language models, AI Open, 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S2666651021000310)]
* **Ernie 3.0:** Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv, 2021. [[Paper](https://arxiv.org/abs/2107.02137)]
* **JURASSIC-1:** Technical Details and Evaluation, White Paper, 2021.
* **HyperCLOVA:** What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers, arXiv, 2021. [[Paper](https://arxiv.org/abs/2109.04650)]
* **Yuan 1.0:** Large-scale pre-trained language model in zero-shot and few-shot learning, arXiv, 2021. [[Paper](https://arxiv.org/abs/2110.04725)]
* **Gopher:** Scaling language models: Methods, analysis & insights from training gopher, arXiv, 2021. [[Paper](https://arxiv.org/abs/2112.11446)]
* **Ernie 3.0 titan:** Exploring larger-scale knowledge enhanced pre-training for language understanding and generation, arXiv, 2021. [[Paper](https://arxiv.org/abs/2112.12731)]
* **Gpt-neox-20b:** An open-source autoregressive language model, arXiv, 2022. [[Paper](https://arxiv.org/abs/2204.06745)]
* **Opt:** Open pre-trained transformer language models, arXiv, 2022. [[Paper](https://arxiv.org/abs/2205.01068)]
* **Bloom:** A 176b-parameter open-access multilingual language model, arXiv, 2022. [[Paper](https://arxiv.org/abs/2211.05100)]
* **Glam:** Efficient scaling of language models with mixture-of-experts, ICML, 2022. [[Paper](https://proceedings.mlr.press/v162/du22c.html)]
* **MT-NLG:** Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model, arXiv, 2022. [[Paper](https://arxiv.org/abs/2201.11990)]
* **Chinchilla:** Training compute-optimal large language models, arXiv, 2022. [[Paper](https://arxiv.org/abs/2203.15556)]
* **Alexatm 20b:** Few-shot learning using a large-scale multilingual seq2seq model, arXiv, 2022. [[Paper](https://arxiv.org/abs/2208.01448)]
* **Palm:** Scaling language modeling with pathways, arXiv, 2022. [[Paper](https://arxiv.org/abs/2204.02311)]
* **U-Palm:** Transcending scaling laws with 0.1% extra compute, arXiv, 2022. [[Paper](https://arxiv.org/abs/2210.11399)]
* **Ul2:** Unifying language learning paradigms, ICLR, 2022. [[Paper](https://openreview.net/forum?id=6ruVLB727MC)]
* **Glm-130b:** An open bilingual pre-trained model, arXiv, 2022. [[Paper](https://arxiv.org/abs/2210.02414)]
* **Llama:** Open and efficient foundation language models, arXiv, 2023. [[Paper](https://arxiv.org/abs/2302.13971)]
* **PanGu-Sigma:** Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing, arXiv, 2023. [[Paper](https://arxiv.org/abs/2303.10845)]
### Coding
* **Codegen:** An open large language model for code with multi-turn program synthesis, arXiv, 2022. [[Paper](https://arxiv.org/abs/2203.13474)]
* **Codex:** Evaluating large language models trained on code, arXiv, 2021. [[Paper](https://arxiv.org/abs/2107.03374)]
* **Alpha Code:** Competition-level code generation with alphacode, Science, 2022. [[Paper](https://www.science.org/doi/abs/10.1126/science.abq1158)]
* **Codet5+:** Open code large language models for code understanding and generation, arXiv, 2023. [[Paper](https://arxiv.org/abs/2305.07922)]
* **StarCoder:** may the source be with you!, arXiv, 2023. [[Paper](https://arxiv.org/abs/2305.06161)]
### Scientific Knowledge
* **Galactica:** A large language model for science, arXiv, 2022, [[Paper](https://arxiv.org/abs/2211.09085)]
### Dialog
* **Lamda:** Language models for dialog applications, arXiv, 2022. [[Paper](https://arxiv.org/abs/2201.08239)]
* **Sparrow:** Improving alignment of dialogue agents via targeted human judgements, arXiv, 2022. [[Paper](https://arxiv.org/abs/2209.14375)]
### Finance
* **Bloomberggpt:** A large language model for finance, arXiv, 2023. [[Paper](https://arxiv.org/abs/2303.17564)]
* **XuanYuan 2.0:** A Large Chinese Financial Chat Model with Hundreds of Billions Parameters, arXiv, 2023. [[Paper](https://arxiv.org/abs/2305.12002)]
## Fine-tuned LLMs

### Instruction-tuned with Manually Created Datasets
* **T0:** Multitask prompted training enables zero-shot task generalization, arXiv, 2021. [[Paper](https://arxiv.org/abs/2110.08207)]
* **mT0:** Crosslingual generalization through multitask fine-tuning, arXiv, 2022. [[Paper](https://arxiv.org/abs/2211.01786)]
* **Tk-Instruct:** Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks, arXiv, 2022. [[Paper](https://arxiv.org/abs/2211.01786)]
* **Opt-iml:** Scaling language model instruction meta learning through the lens of generalization, arXiv, 2022. [[Paper](https://arxiv.org/abs/2212.12017)]
* **Flan:** Scaling instruction-finetuned language models, arXiv, 2022. [[Paper](https://arxiv.org/abs/2210.11416)]
* **The CoT Collection:** Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning, arXiv, 2023. [[Paper](https://arxiv.org/abs/2305.14045)]
* From zero to hero: Examining the power of symbolic tasks in instruction tuning, arXiv, 2023. [[Paper](https://arxiv.org/abs/2304.07995)]

### Instruction-tuned with LLMs Generated Datasets


