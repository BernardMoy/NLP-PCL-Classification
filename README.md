# Fine-tuning a RoBERTa Baseline Model on Detecting Patronizing and Condescendig Language (PCL) towards Vulnerable Communities

This repository describes an approach that improves the F1 score from 0.577 to 0.630, on performing binary classification towards PCL. The following modifications were used to improve the model performance:

- Oversampling the minority class
- Adding contextual information (Keyword, country code) to the text
- Use of model ensembles by combining a BERT and RoBERTa model

## Folder Structure

```text
root/
│
├── BestModel/
│   └── ensemble.ipynb
│
├── data/
│   ├── dontpatronizeme_pcl.tsv
│   ├── train_semeval_parids-labels.csv
│   ├── dev_semeval_parids-labels.csv
│   ├── task4_test.tsv
│   └── ...
│
├── evaluation/
│   ├── baseline.txt
│   ├── bert_ensemble.txt
│   ├── evaluation.ipynb
│   ├── final.txt
│   ├── only_oversample.txt
│   ├── oversample_context_cr.txt
│   └── roberta_ensemble.txt
│
├── models_implementation/
│   ├── baseline.ipynb
│   ├── bert_ensemble.ipynb
│   ├── only_oversample.ipynb
│   ├── oversample_context_cr.ipynb
│   └── roberta_ensemble.ipynb
│
├── analysis.py
├── dev.txt
├── test.txt
├── report.pdf
└── ...
```

### Description

- `BestModel/`: Contains `ensemble.ipynb` that loads improved trained models and create the ensemble model.
- `data/`: Contains the PCL dataset train and test data, and the indices for train / val split.
- `evaluation/`: Contains the `evaluation.ipynb` file to perform evaluation on different models, and contains labels predicted from different models.
- `models_implementation/`: Contains the main implementation of different approaches
- `models/`: Not in this repository (too large). Contains trained models that can be loaded.
- `analysis.py`: Display exploratory data analysis results of the PCL dataset.
- `dev.txt`: Prediction results for the official dev set (0 for not PCL, 1 for PCL)
- `test.txt`: Prediction results for the official test set.
- `report.pdf`: A report documenting the proposed approach, detailed metrics and error analysis.

### Implemented Approaches

The implemented approaches are described in the `models_implementation/` folder. All of them are trained with batch size = 32 with 5 epochs, where the best model was chosen.

- `baseline.ipynb`: RoBERTa baseline from HuggingFace
- `only_oversample.ipynb`: RoBERTa baseline + Oversampling the minority (positive) class
- `roberta_ensemble.ipynb`: RoBERTa baseline + Oversampling the minority class + adding contextual information (keyword, country code) to the text. Used as the RoBERTa ensemble.
- `oversample_context_cr.ipynb`: RoBERTa baseline + Oversampling the minority class + Coreference resolution. Discarded due to poor performance.
- `bert_ensemble.ipynb`: Similar approach to `roberta_ensemble.ipynb`, but used a BERT base model instead
