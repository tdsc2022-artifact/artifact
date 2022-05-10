## This toolkit is for the paper "How About Bug-Triggering Paths? - Understanding and Characterizing Learning-Based Vulnerability Detectors"



**This toolkit contains implementations of all learning-based vulnerability detection methods used in the paper**

###  For popular vulnerability detection methods, we build a unified framework. You can adapt to different datasets and train different models only by modifying the parameter settings。

| methods         | paper                                                        |
| --------------- | ------------------------------------------------------------ |
| code2seq        | code2seq: Generating sequences from structured representations of code |
| code2vec        | code2vec: Learning distributed representations of code       |
| DeepWukong      | DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network |
| μVulDeePecker   | μVulDeePecker: A Deep Learning-Based System for Multiclass Vulnerability Detection |
| VulDeePecker    | The Network and Distributed System Security Symposium        |
| SySeVr          | SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities |
| ReVeal          | Deep Learning based Vulnerability Detection: Are We There Yet? |
| Vgdetector      | Static Detection of Control-Flow-Related Vulnerabilities Using Graph Embedding |
| token embedding | Automated vulnerability detection in source code using deep representation |
| IVDetect      | Vulnerability Detection with Fine-Grained Interpretations    |
| VulDeeLocator | VulDeeLocator: A Deep Learning-based Fine-grained Vulnerability Detector |
| ICVH          | Information-theoretic Source Code Vulnerability Highlighting |
| VELVET        | VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements |

*The framework contains four directories: config/ , models/ , preprocessing/ , utils/*

 `config: All parameter settings. `

`models: Models are all implemented using pytorch lightning.`

`preprocessing: Data Preprocessing for Partial Methods`.

`utils: Commonly used tool functions`

`train.py: Entry point for all model training steps`

### Part of the work uses [Joern](https://joern.io/) for program analysis and extracts program slices

*We implemented Joern-based slicing methods for Deepwukong, Vuldeepecker, SySevr. But using an old version of Joern which can generate PDG's node.csv and edge.csv.*

`joern_slicer: slicing methods. If you want to use this part of the code, you need a Joern version that can generate a csv file of PDG`.

*you can find it here [old Joern](https://github.com/ives-nx/dwk_preprocess/tree/main/joern_slicer/joern)*


