# Twitter Topic Modeling

Project by Leo Corelli

[Link to project website](https://share.streamlit.io/leocorelli/twitter-topic-modeling/main/streamlit.py)

[Link to demo video](https://duke.box.com/s/uhq7hcukm5c1tb20kmt7kfxrqi7rshpy)

<p align="center">
  <img src="https://github.com/leocorelli/twitter-topic-modeling/blob/main/images/Twitter-logo.png" width="220" /> 
</p>

## About

This project pulls in real-time data from the Twitter API and uses a sentence transformer model, UMAP (non-linear dimension reduction algorithm) , density-based clustering, and class-based TF-IDF to cluster tweets on a specific topic into relevant subtopics for enhanced and actionable insight.

## How it works

[BERTopic package](https://maartengr.github.io/BERTopic/index.html)

**1. Transformers**
  - They’re amazing at wide range of NLP tasks
  - BERT (a pre-trained language model) *understands*
    - Extracts different embeddings based on the context of the word

**2. Reduce dimensionality using UMAP**
  - Many clustering algorithms handle high dimensionality poorly
  - “Keeps a significant portion of the high-dimensional local structure in lower dimensionality”

**3. HDBSCAN clustering (Density-based clustering)**
  - Does not require a pre-determined set number of clusters
  - Maintains a lot of local structure (doesn’t force into circular clusters)
  - Does not force data points to clusters as it considers them outliers

**4. Topic Creation with c-TF-IDF (class based TF-IDF)**
  - Normal TF-IDF: consider each document as its own document
  - Treat all documents in a single cluster as one document
    - Then apply TF-IDF across all the clusters
    - Result: what words make each cluster unique!


## How to run
- Go to [project link](https://share.streamlit.io/leocorelli/twitter-topic-modeling/main/streamlit.py)
- Type term into the search bar and click enter!

## Credit
A special thank you to Maarten Grootendorst and his amazing work in creating the BERTopic library. Additionally, a special thank you to Jon Reifschneider of Duke University for his amazing teaching this semester in AIPI 540.
```
@misc{grootendorst2020bertopic,
  author       = {Maarten Grootendorst},
  title        = {BERTopic: Leveraging BERT and c-TF-IDF to create easily interpretable topics.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.9.4},
  doi          = {10.5281/zenodo.4381785},
  url          = {https://doi.org/10.5281/zenodo.4381785}
}
```
