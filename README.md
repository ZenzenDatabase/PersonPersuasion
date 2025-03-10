# Generative Framework for Personalized Persuasion: Inferring Causal, Counterfactual, and Latent Knowledge

# Supplementary Materials

 A. Theoretical Foundations, Motivations, and Justifications of the Methods. [Link](#A.-Theoretical-Foundations,-Motivations,-and-Justifications-of-the-Methods.)

 B. Reproducibility [Link](#B.-Reproducibility)

 C. Statistical Tests of Significance [Link](#C.-Statistical-Tests-of-Significance)

 D. Codes [Link](#D.-Codes)

## A. Theoretical Foundations, Motivations, and Justifications of the Methods.(#sample-section)

Our choice of methods is driven by their sufficiency in testing our hypothesis—the ideal persuasive system actions emerge as byproducts of a generative framework that leverages causal, counterfactual, and latent factor (hidden personality traits and unobserved noise) components to improve persuasion outcomes. We aim to validate the added effect of the hypothesis-related components when compared, ceteris paribus, to the recent generative framework of (Zeng et al., Persuasive 2024), which we treated as the "vanilla"/control (baseline) version. This means that, apart from the hypothesis-supporting components, we adopted the parts of the control version. In effect, we adopted a baseline that provided a simple, straightforward, and effective starting point for comparison that helped prove our hypothesis and demonstrate its benefits. Furthermore, while our results clearly evidence the potential of our generative framework for optimization over the baseline, more can be done to achieve full optimization. Given that the components of our framework are modularized, future work may explore alternative methods to enable further optimization.

We detail below our motivations and justifications for our choice of methods:

#### BERT Model
First and foremost, the dialogue utterances need to be embedded into vector representations, capturing the contextual relationships and nuanced meanings in the natural language. These vector representations would become the primitives of the user states and system actions that are passed throughout the pipeline. While specialized models like GPT-based architectures or Transformer-based dialogue systems (e.g., ChatGPT, T5, or LaMDA) are often more preferred, BERT is a strong baseline in the event that we need to see the effect of using another method. Furthermore, BERT is useful for embedding utterances in dialogue because it provides contextualized word representations, allowing words to adapt their meaning based on conversation contexts. Its self-attention mechanism captures long-range dependencies, making it effective for understanding multi-turn interactions like in our use case.

#### Turn-based Persuadee Personality Prediction Model (TP3M)
To leverage the effect of latent psychological constructs on counterfactual inferencing, it was impactful for us to ask the likelihood of saying a word relates to how neurotic or agreeable someone is, i.e., how can we predict latent personality traits from just what is being said, believing that these hidden factors affect the state transition dynamics? This model is developed to predict personality traits from the ongoing dialogue. This helps the system tailor its responses to individual preferences and dispositions. Furthermore, TP3M is unbiased as it does not overly rely on personality stereotypes, some of which may not be true, hence, can be harmful. Instead, the TP3M dynamically fine-tunes its personality trait estimates based on the utterances in the ensuing dialogue.

#### Fine-tuned GPT-2 Models
It is known that GPT-2 has been trained on a vast dataset, which enables it to generate responses based on a wide range of topics without requiring extensive fine-tuning. This makes GPT-2 also useful for general-purpose dialogue agents. It is one of the early large language models (LLMs) but sufficient for us given that it is more accessible. Other LLMs can be explored based on their availability and cost (heads up, acquiring a license can be very expensive!).

#### Bidirectional Conditional Generative Adversarial Network (BiCoGAN) vs. Kernel Quantile Regression (KQR)
Both methods for counterfactual inference estimate the structural causal model from the observational data. However, while BiCoGAN estimates the noise explicitly, KQR does not need to. We want to know their differentiating effects following the hypothesis.

#### Greedy Relaxation of the Sparsest Permutation (GRaSP)
We looked for a method more scalable than brute-force permutation search, and one that would significantly reduce the computational burden without compromising accuracy. Furthermore, we realized that given the predefined set of strategy constructs in the data we used, we expected it to result in a sparse causal graph. This makes GRaSP particularly effective since it prioritizes finding the minimal set of causal edges that explain the data. In other words, it explicitly optimizes for sparsity in the causal ordering.

#### Retrieval-based Model (RBM)
Successfully validated in ([^Tran2022]), we use this model because it efficiently selects contextually appropriate counterfactual actions by uncovering and leveraging the underlying causal structure in the dialogues at the utterance-strategy level. By mapping the persuadee strategy to the persuader strategy through the causal graph, the RBM could ensure that the counterfactual actions align with the persuasion outcomes.

#### Dueling Double Deep Q-Network (D3QN)
This model reduces overestimation bias in Q-values. Its dual Q-networks ensure stable training and more accurate decision-making.


[^Tran2022]: Tran et al., *"How to ask for donations? Learning user-specific persuasive dialogue policies through online interactions,"* UMAP 2022.[supporting link](https://doi.org/10.1145/3503252.35313)



## B. Reproducibility
section*{Reproducibility}

We discuss here the details necessary to reproduce the various aspects of our generative framework.

### Real-world Dataset

To prove our hypothesis and validate the performance of our generative framework, we used the ``Persuasion For Good'' (P4G) data corpus (https://convokit.cornell.edu/documentation/persuasionforgood.html). P4G is a collection of online conversations generated by Amazon Mechanical Turk workers, where one participant (the Persuader) tries to convince the other (the Persuadee) to donate to a charitable organization called Save the Children. This dataset contains 1,017 conversations, along with demographic data and responses to psychological surveys from users. Of the total number of dialogues, 300 have per-sentence human annotations of dialogue acts that pertain to the persuasion setting and sentiment.

###Extraction of the Pertinent Aspects in the Data

We used all 1,017 dialogues of P4G in our experiments. ANNSET is the subset of P4G that contains the annotated data, where each utterance is labeled with a utterance strategy. The rest of P4G lacks such annotations, which is why we leveraged GPT-2 to predict the strategies for the unannotated utterances. 

As for the psychological constructs, we are only interested at this time (other constructs can be used later on) with the Big Five personality traits. Known as the Big Five because of its five personality trait dimensions, i.e., OCEAN, which stands for Openness, Conscientiousness, Extroversion, Agreeableness, and Neuroticism. Variations in this small set of broad dispositional traits linked to social life often predict significant differences in behavior, especially when behavior is considered across different situations.     

In P4G, the BIg Five are represented as continuous values between 1 and 5 (e.g., Open: 3.8, Conscientious: 4.4, Extrovert: 3.8; Agreeable: 4.0, and Neurotic: 2.2). These trait inventory scores were computed from the participants’ self-ratings of trait-descriptive questions, and the responses for each trait were averaged. We used these final ratings to predict the personalized influence of the traits on the state transition dynamics. 
 
### Data Preprocessing

We preprocessed the data through the following steps. Firstly, consecutive utterances from the same role, persuader or persuadee, were combined. We then performed tokenization, specifically, we used the ``BertTokenizerFast'' from the pretrained ``google-bert/bert-base-uncased''. This ensured the sequences are either padded or truncated to a maximum length of 512 tokens. Each dialogue is then converted into BERT embeddings following the implementation details provided in https://huggingface.co/docs/transforodeers/en/model_doc/bert.Finally, we transformed the donation amounts by scaling each user donation to a given range (Min to Max).  

### Model Implementation, Training and Testing

We implemented our models using PyTorch and trained them on a GeForce RTX 3080 GPU (10G). The model configurations used in the experiments are as follows: (1) TP3M: 1024 hidden units, a batch size of 64, learning rate of 0.0001, and 100 epochs. (2) BiCoGAN: 100 hidden units, a batch size of 100, learning rate of 0.0001, and 10 epochs. (3) RB: 256 hidden units, a batch size of 64, learning rate of 0.0001, and 1,000 epochs. D3QN: 256 hidden units, a batch size of 60, learning rate of 0.001, and 20 epochs. (4) KQR: A batch size of 200 with a learning rate of 0.1 for ParentNet and 0.001 for ChildNet.
For BiCoGAN, we set the dimension of the noise term to be the same as BERT's embedding of 768. 
To reduce potential bias from outliers and ensure a more balanced training of the models, we set any donation exceeding \$10 to be \$10. 

We generated 50 counterfactual dialogues using different sets of counterfactual actions. Each dialogue showcases flexible-length exchanges between EE and ER, with alternating utterances. We did an 80/20-split of the data for training/testing. All models were optimized using Adam with a 0.0001 learning rate. We also set D3QN's discount factor $\gamma$ to 0.9. For the KQR, we use a decay factor (controls the smoothness of the kernel) of 0.9 for the ParentNet and apply a QP solver during training for the ChildNet. Lastly, to ensure reproducibility, we ``controlled'' for randomness using ``random.seed(42)''.

## C. Statistical Tests of Significance

### Statistical Validation for Claims: persuasion outcome.

#### p-value (T-test)
To evaluate the effectiveness of our proposed method in improving donation
outcomes, we conducted Welch’s t-tests to compare different strategies. The
results indicate statistically significant improvements when using latent variable
integration with causal discovery compared to baseline and random strategies.

** Model Comparison **

*** BiCoGAN-based Models ***

- **CD_Latent_BiCoGAN vs. CD_BiCoGAN**: \( 8.18 * 10^{-66} \)
- **CD_Latent_BiCoGAN vs. Random_Latent_BiCoGAN**: \( 6.51 * 10^{-294} \)
- **CD_Latent_BiCoGAN vs. Random_BiCoGAN**: \( 1.87 * 10^{-195} \)

*** KQR-based Models ***

- **CD_Latent_KQR vs. CD_KQR**: \( 1.66 * 10^{-75} \)
- **CD_Latent_KQR vs. Random_Latent_KQR**: \( 1 * 10^{-300} \)
- **CD_Latent_KQR vs. Random_KQR**: \( 1 * 10^{-300} \)

#### Confidence Intervals

*** BiCoGAN-based Models *** 

- **BiCoGAN + Latent + CD vs. BiCoGAN + CD**:  
  The difference in means between BiCoGAN + Latent + CD and BiCoGAN + CD is statistically significant, with the 95% confidence interval for the difference being **(1.46, 1.93)**. This indicates that the latent variable integration significantly improves donation outcomes compared to the BiCoGAN + CD method.  

- **BiCoGAN + Latent + CD vs. BiCoGAN + Random**:  
  The difference in means between BiCoGAN + Latent + CD and BiCoGAN + Random is also statistically significant, with the confidence interval **(3.75, 4.12)**. This shows a substantial improvement in donation amounts when causal discovery is combined with latent variables.  

- **BiCoGAN + Latent + CD vs. Random BiCoGAN**:  
  The confidence interval **(2.85, 3.25)** indicates a statistically significant improvement in donation amounts for BiCoGAN + Latent + CD compared to Random BiCoGAN.  

*** KQR-based Models ***  

- **KQR + Latent + CD vs. KQR + CD**:  
  The difference in means between KQR + Latent + CD and KQR + CD is statistically significant, with the confidence interval **(0.94, 1.17)**. This suggests a notable improvement in donations when integrating latent variables with causal discovery.  

- **KQR + Latent + CD vs. Random KQR**:  
  The confidence interval **(3.78, 3.80)** shows a very small but statistically significant difference, suggesting that KQR + Latent + CD outperforms Random KQR in terms of donation outcomes.  

- **KQR + Latent + CD vs. Random KQR with Latent**:  
  The difference between KQR + Latent + CD and Random Latent KQR is significant, with a confidence interval **(2.99, 3.07)**. This indicates a clear advantage for the model that integrates causal discovery with latent variables.  

The results demonstrate that integrating latent personality and causal discovery significantly enhances donation outcomes in both BICOGAN and KQR models compared to random strategies, with statistical tests confirming the improvements are not due to chance. This highlights the effectiveness of causal discovery in persuasion strategies.

## D. Codes

Refer to the [How_to_use_trained_models.ipynb](./How_to_use_trained_models.ipynb) for instructions on using the trained models with example inputs.
We shall make the codes and libraries be available if our paper is accepted.
