# Modeling Political Distances

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

## Results

Our results indicate that the Transformer-based neural network outperforms the other baseline
models, achieving an area under the curve (AUC) of 0.92 on the VOX-PODEMOS test dataset.
This slight improvement by our simple transformer architecture suggests that employing a more
complex model, such as a large language model (LLM), could further enhance our results.

____________

### Report Poster is available on the repository under "DL_Final_Poster_Breier_Mirabent"
### Transformer model is available under "Transformer_Model" notebook
