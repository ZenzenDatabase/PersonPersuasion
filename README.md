# Generative Framework for Personalized Persuasion: Inferring Causal, Counterfactual, and Latent Knowledge

# Supplementary Materials

 A. Theoretical Foundations, Motivations, and Justifications of the Methods. [Link Text](#A. Theoretical Foundations, Motivations, and Justifications of the Methods.)

 B. Reproducibility [Link Text](#B. Reproducibility)

 C. Statistical Tests of Significance [Link Text](#C. Statistical Tests of Significance)

 D. Codes [Link Text](#D. Codes)

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

## C. Statistical Tests of Significance

## D. Codes
