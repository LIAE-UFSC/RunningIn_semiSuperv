# Supervised learning for [running-in](https://en.wikipedia.org/wiki/Break-in_(mechanical_run-in)) detection in hermetic alternative compressors

## Description

Analysis of supervised learning method to aid in the detection of the running-in state in hermetic alternative refrigerant compressors.

This project is designed for analysing the performance of different supervised learning methods according to different metrics, as proposed in [[1]](#1).

This repository contains examples of labeled data, with labels extracted using the method proposed in [[2]](#2).

## Available methods

- [x] Logistic regression
- [x] Linear SVM
- [x] Polynomial SVM
- [x] RBF SVM
- [x] k-nearest neighbors
- [x] Random forest

## Available metrics

- [x] Matthews correlation coefficient (MCC)
- [x] ROC-AUC
- [x] F-score
- [x] F<sub>β</sub>-score

## Getting Started

### Pre-requisites

* Git

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/nicolasantero/compressor-breakin-kmeans-clustering.git
   ```
   
 2. Installing Packages
   ```sh
   pip install requirements.txt
   ```
   
<p align="right">(<a href="#top">Início</a>)</p>

### Examples

See 'analise_modelos.py'.

## References

<a id="1">[1]</a> 
Thaler, G. (2021). [_Desenvolvimento de métodos não invasivos para avaliação do processo de amaciamento de compressores herméticos alternativos_](https://repositorio.ufsc.br/handle/123456789/230918) [Development of non-invasive methods for evaluating the running-in process in reciprocating hermetic compressors] (Master's thesis, Federal University of Santa Catarina, Florianópolis, Brazil).

<a id="2">[2]</a> 
Thaler, G., Nunes, N. A., Nascimento, A. S. B. de S., Pacheco, A. L. S., Flesch, R. C. C. (2021). [Aplicação de aprendizado não supervisionado para identificação não destrutiva do amaciamento em compressores](https://doi.org/10.20906/sbai.v1i1.2611) (Application of unsupervised learning for non-destructive running-in identification in compressors). Proceedings of SBAI 2021, Brazil, 460–466.
# projeto_prh
# LIAE_Amaciamento
# LIAE_Amaciamento
