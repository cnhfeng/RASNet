# "RASNet: Recurrent Aggregation Neural Network for Safe and Efficient Drug Recommendation"

# Results

* Results of the paper:

  ``` python
  DDI: 0.0599(0.0009) Ja: 0.5401 (0.0021) PRAUC: 0.7882 (0.0025) F1: 0.6931 (0.0019) 
  ```

## Folder Specification

* ```data/output/```
  * **ddi_A_final.pkl**: ddi adjacency matrix
  * **ehr_adj_final.pkl**: used in GAMENet baseline (if two drugs appear in one set, then they are connected)
  * **records_final.pkl**: The final diagnosis-procedure-medication EHR records of each patient
  * **voc_final.pkl**: diag/prod/med index to code dictionary
* ```src/```
  * **model.py**: Code for model definition.
  * **util.py**: Code for metric calculations and some data preparation.
  * **test.py**: Code for reproducing the paper results
  * **layer.py**

We cannot distribute the whole MIMIC-III data https://physionet.org/content/mimiciii/1.4/, then please download the dataset by yourself.
## Package Dependency

* first, install the conda environment

  ```python
  conda create -n RASNet python=3.8
  conda activate RASNet
  ```

* then, in RASNet environment, install the following package
  ```py
  pip install scikit-learn, dill
  ```
* Finally, install other packages if necessary
  ```py
  pip install [xxx]
  ```
  Here is a list of reference versions for all package
  ```py
  pandas: 1.5.2
  dill: 0.3.6
  torch: 2.0.1
  scikit-learn: 1.2.0
  numpy: 1.23.4
  ```

##  Run 

Run test.py to reproduce the results of the paper.

```python
cd src
python test.py
```
## Citation
Related papers：- https://www.sciencedirect.com/science/article/pii/S0950705124006890

If the code and the paper are useful for you, it is appreciable to cite our paper:
```python
@article{zhu2024rasnet,
  title={RASNet: Recurrent aggregation neural network for safe and efficient drug recommendation},
  author={Zhu, Qiang and Han, Feng and Yang, Huali and Liu, Junping and Hu, Xinrong and Wang, Bangchao},
  journal={Knowledge-Based Systems},
  pages={112055},
  year={2024},
  publisher={Elsevier}
}
```
Partial credit to previous reprostories:

- https://github.com/sjy1203/GAMENet
- https://github.com/ycq091044/SafeDrug
- https://github.com/yangnianzu0515/MoleRec
