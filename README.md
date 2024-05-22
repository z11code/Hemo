# Hemo

Hemolytic peptides are therapeutic peptides that damage red blood cells. However, therapeutic peptides used in medical treatment must exhibit low toxicity to red blood cells to achieve the desired therapeutic effect. Therefore, accurate prediction of the hemolytic activity of therapeutic peptides is essential for the development of peptide therapies. In this study, a multi-feature cross-fusion model, HemoFuse, for hemolytic peptide identification is proposed.

Python=3.8，torch=1.8.0，pandas=1.1.3，numpy=1.18.0。

D1_train_pos.fa is the training positive sample of the dataset 1, and D1_train_neg.fa is the training negative sample of the dataset 1. D1_test is the corresponding independent test dataset. Others are similar.

main.py is the model HemoFuse used in the study. You can see its internals and use it to train your data.

predict.py is used to test the performance of the model.

D1.pt，D2.pt，D3.pt，D4.pt，integral.pt is our already trained model.
