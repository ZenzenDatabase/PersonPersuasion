# PersonPersuasion

## Model Selection Criteria

- **BERT**: We first needed to embed dialogue utterances into vector representations while capturing contextual relationships and nuanced meanings in natural language. These vector representations serve as the building blocks for the states and actions throughout the pipeline. While specialized models like GPT-based architectures or Transformer-based dialogue systems (e.g., ChatGPT, T5, or LaMDA) are often preferred, BERT provides a strong baseline for comparison. Additionally, BERT is particularly effective for embedding utterances in dialogue due to its ability to provide contextualized word representations, allowing words to adapt their meaning based on conversation context. Its self-attention mechanism captures long-range dependencies, making it well-suited for understanding multi-turn interactions.

- **TP3M**: To explore how latent psychological constructs influence counterfactual inference, we investigated how the likelihood of using certain words relates to personality traits such as neuroticism or agreeableness. Our model predicts latent personality traits dynamically from ongoing dialogue rather than relying on personality stereotypes, which can be misleading or harmful. This approach ensures that the system fine-tunes its personality trait estimates based on conversational context. Our experiments demonstrated that focusing on one-turn interactions yielded the best performance across various evaluation metrics (MSE, RMSE, MAPE, R², and MAE).

- **GPT-2**: As an early large language model (LLM), GPT-2 has been trained on a vast dataset, enabling it to generate responses on a wide range of topics without requiring extensive fine-tuning. Given its accessibility, GPT-2 serves as a suitable model for our needs. However, other LLMs may be explored in the future based on availability and cost, as acquiring licenses for some models can be prohibitively expensive.

- **BiCoGAN vs. KQR**: Both methods estimate the structural causal model from observational data. BiCoGAN explicitly models noise, while KQR does not require explicit noise estimation. We compare these models to analyze their differing effects based on our hypothesis.

- **GRaSP**: We sought a method more scalable than brute-force permutation search that could significantly reduce computational burden without compromising accuracy. Given the predefined strategy constructs in our dataset, we expected a sparse causal graph. GRaSP is particularly effective in this setting, as it explicitly optimizes for sparsity in causal ordering by identifying the minimal set of edges that explain the data.

- **RB**: Successfully validated in [1], this model efficiently selects contextually appropriate counterfactual actions by leveraging causality at the strategy level. By mapping persuadee (EE) strategies to persuader (ER) strategies via the causal graph, RB ensures that counterfactual actions align with intended outcomes.

- **D3QN**: We selected D3QN to mitigate overestimation bias in Q-values. Its dual Q-networks provide more stable training and improve decision-making accuracy.

## References

[1] Tran et al. *"How to ask for donations? Learning user-specific persuasive dialogue policies through online interactions,"* UMAP 2022.

---

The reviewer also noted that our work **"may be better seen as a proof of concept regarding the underlying hypothesis than a fully optimized result."**  
Our results provide strong evidence supporting our hypothesis: causal discovery, in combination with hidden psychological factors and unobserved noise influencing state-action transition dynamics, enables a principled counterfactual inference approach that leads to improved outcomes.
