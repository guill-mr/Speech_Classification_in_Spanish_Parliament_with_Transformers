# Modeling Political Distances using Transformer Based Neural Network

### 27th June, 2024
Mathieu Breier & Guillem Rubiat Mirabent
__________

## Introduction
The spanish political landscape, once dominated by the Spanish Socialist Workers’ Party (PSOE)
and the Popular Party (PP), fragmented since the mid-2010s. This shift began around 2014-2015
with the emergence of new parties. Podemos, founded in 2014, positioned itself as a radical
leftist alternative amid discontent from the financial crisis. Ciudadanos, a liberal centrist party,
also rose around the same time, advocating for national unity and economic liberalism. Vox, a far-
right party, entered the national parliament in April 2019 and focuses on national unity, opposing
Catalan independence, criticizing immigration, and rejecting political correctness.

These changes have created a more polarized and complex political environment in Spain. Coali-
tion governments are now more common, and traditional parties must negotiate compromises.

In this context, we aim to explore party lines and ideologies using a transformer neural networks
architecture, seeking insights into how Spanish politicians express their beliefs. We will investi-
gate whether a deep learning model can accurately distinguish speeches from different parties.
This study builds upon the work of A. Bennett, M. Handt, and G. Mirabent on speech clustering
in Spanish politics using Natural Language Processing techniques. The scope of this study could
extend to identifying political ideologies in various speeches, thereby enhancing support for var-
ious economic analyses

## Data

The data comes from an open source project called ParlaMint. The project team at ParlaMint has
worked to create extensive metadata for each of the speeches, including speaker names, party
affiliations, gender, age and position. We selected the Spanish corpus for our analysis.

The Spanish corpus consists of 32,551 speeches from the Spanish parliament. The speeches span
8 years from 2015 to 2023, and includes 5 legislative terms. In total, there were over 50 parties
in the data including subgroups of the two major parties we are interested in, PP and PSOE.

## Results

Our results indicate that the Transformer-based neural network outperforms the other baseline
models, achieving an area under the curve (AUC) of 0.92 on the VOX-PODEMOS test dataset.
This slight improvement by our simple transformer architecture suggests that employing a more
complex model, such as a large language model (LLM), could further enhance our results.

____________

### Report Poster is available on the repository under "DL_Final_Poster_Breier_Mirabent"
### Transformer model is available under "Transformer_Model" notebook
### Pre-Processed Data is located in the zip.file
