# Thesis Defense Presentation Plan

## Presentation objective

The presentation should explain the problem clearly within the first two minutes and then focus on the thesis's central experimental story:

1. Which GNN architecture retrieves the correct answer entities most effectively?
2. How much can the evidence subgraph be reduced without losing relevant information?
3. Why does preserving a correct entity in the context still not guarantee that the LLM will return it?

The research questions are introduced near the beginning and answered directly near the end. The main presentation contains 21 slides and targets approximately 18 minutes and 10 seconds, leaving a safety margin within the 20-minute limit.

## Part I: Problem and research questions

### Slide 1: Наслов

**Target time:** 0:15

**Purpose:** Introduce the thesis and presenter.

**Include:**

- Thesis title
- Candidate: Виктор Костадиноски
- Mentor: проф. д-р Соња Гиевска
- Faculty and defense date

**Presentation note:** Keep this slide minimal. Introduce the topic in one sentence and continue immediately.

### Slide 2: Од прашање до одговор

**Target time:** 0:45

**Purpose:** Make the task understandable through one concrete example.

**Include:**

- One representative WebQSP question
- The seed entity identified in the question
- A simplified noisy local knowledge graph
- The expected answer entity
- A visual indication that the answer must be found together with the facts connecting it to the question

**Suggested example:** „Во која временска зона се наоѓа Кливленд, Охајо?“ with „Кливленд“ as the seed entity and „Eastern Time Zone“ as the answer.

**Main message:** The system must identify the answer among many graph entities and preserve enough relational evidence to justify it.

### Slide 3: Архитектура на системот

**Target time:** 0:50

**Purpose:** Explain the complete task before introducing technical details.

**Include:**

- Question and local graph
- GNN-based entity retrieval
- Ranked entities-candidates
- Evidence-subgraph construction
- LLM-based final answer generation

**Visual:** Use a presentation-adapted version of the system overview figure.

**Main message:** The system contains three measurable transitions: retrieval, evidence construction and answer generation.

By the end of this slide, approximately 1 minute and 50 seconds into the talk, the audience should understand what the system does.

### Slide 4: Сродни пристапи и истражувачка празнина

**Target time:** 0:50

**Purpose:** Position the thesis relative to the most relevant previous work.

**Include:**

- GNN-RAG: GNN retrieval followed by shortest-path evidence and LLM reasoning
- G-Retriever: PCST-based selection of compact evidence from textual graphs
- The remaining need for a controlled comparison of substantially different GNN architectures, evidence-construction strategies and language models in one staged framework

**Main message:** Previous work establishes the core ideas, while this thesis evaluates how the choices in all three stages affect retrieval, context and final answer quality.

### Slide 5: Истражувачки прашања

**Target time:** 0:40

**Purpose:** Establish the questions that organize the rest of the presentation.

**Include:**

1. Како изборот на архитектура на граф невронска мрежа влијае врз рангирањето и опфатеноста на точните одговори?
2. Како стратегијата за конструкција на доказниот подграф влијае врз неговата компактност и врз зачувувањето на релевантните информации?
3. Како изборот на голем јазичен модел влијае врз конечната точност и искористувањето на достапниот доказен контекст?

**Presentation note:** Explain that the questions will be revisited and answered on Slide 20.

## Part II: Methodology

### Slide 6: Податочно множество и експериментална поставеност

**Target time:** 0:45

**Purpose:** Provide only the experimental context required to interpret the results.

**Include:**

- WebQSP questions and question-specific local graphs built from Freebase triples
- 2,848 training examples
- 1,889 combined validation and test examples before missing-gold filtering
- Three random seeds: 42, 1337 and 2026
- Candidate threshold of 0.7
- A minimum of 10 and a maximum of 15 candidates
- The same candidate-selection and evaluation procedure for every architecture

**Main message:** The comparison isolates architecture choice by keeping the dataset and evaluation procedure fixed.

### Slide 7: Архитектури за пребарување

**Target time:** 0:25

**Purpose:** Orient the audience before discussing the model groups.

**Include:**

- GraphSAGE
- Advance GraphSAGE
- R-GCN
- HGT
- ReaRev
- NBFNet

**Visual:** Present the architectures as a conceptual progression from neighborhood aggregation to relation-aware processing, adaptive reasoning and question-conditioned path propagation.

### Slide 8: GraphSAGE и Advance GraphSAGE

**Target time:** 0:35

**Purpose:** Explain the baseline and its question-conditioned extension.

**Include:**

- GraphSAGE aggregates information from neighboring entities
- Edge messages are weighted according to question-relation similarity
- Advance GraphSAGE learns this weighting and additionally uses reverse edges, residual connections, layer normalization and a question-conditioned classifier

**Essential formula:**

$$
\alpha(q,r)=\operatorname{sigmoid}\!\left(\operatorname{MLP}[\tilde q\Vert\tilde r\Vert\tilde q\odot\tilde r]\right)
$$

**Main message:** Advance GraphSAGE learns which relations matter for the current question instead of relying only on fixed cosine similarity.

### Slide 9: R-GCN и HGT

**Target time:** 0:35

**Purpose:** Contrast relation-specific transformations with relation-aware attention.

**Include:**

- R-GCN applies transformations that depend on the relation type
- Basis decomposition controls the number of parameters
- HGT uses relation-dependent multi-head attention to assign different importance to neighboring entities and relations

**Possible formulas:**

$$
h_v^{(l+1)}=\sigma\!\left(W_0^{(l)}h_v^{(l)}+
\sum_{r\in\mathcal R}\sum_{u\in\mathcal N_v^r}
\frac{1}{c_{v,r}}W_r^{(l)}h_u^{(l)}+b^{(l)}\right)
$$

$$
\alpha_{uv}^{r,j}=\operatorname{softmax}_{(u',r',v)\in E}\!\left(e_{u'v}^{r',j}\right)
$$

**Presentation note:** Explain the architectural distinction rather than every symbol.

### Slide 10: ReaRev

**Target time:** 0:40

**Purpose:** Explain adaptive reasoning driven by instructions extracted from the question.

**Include:**

- Token-level question representations
- Multiple learned instructions focused on different parts of the question
- Iterative graph reasoning
- Revision of the instructions based on information found in the preceding iteration

**Essential formula:**

$$
m_{u,r}^{(k,t)}=p_u^{(t)}\operatorname{ReLU}\!\left(W_r^{(t)}r\odot i^{(k)}\right)
$$

**Visual:** Show the cycle: question instructions, graph reasoning, instruction revision, repeated reasoning.

**Main message:** ReaRev can change how it interprets the question as reasoning progresses.

### Slide 11: NBFNet

**Target time:** 0:45

**Purpose:** Explain the path-based architecture that becomes the winning retriever.

**Include:**

- A learned version of Bellman-Ford reasoning
- Question representation initialized at the seed entities
- Question-conditioned relation representations
- Propagation of information along relational paths
- Independent sigmoid scores suitable for questions with multiple correct answers

**Essential formulas:**

$$
h_v^{(0)}=
\begin{cases}
q, & v\in S,\\
0, & v\notin S
\end{cases}
$$

$$
m_{u,r,v}^{(l)}=h_u^{(l)}\odot w_r^{(l)}(q)
$$

**Main message:** NBFNet learns how each candidate can be reached from the seed entities through paths relevant to the question.

Slides 7 through 11 should take approximately three minutes in total.

### Slide 12: Метрики за пребарување

**Target time:** 0:45

**Purpose:** Explain how retrieval quality is measured.

**Explain without formulas:**

- Hits@1 and Hits@10 measure whether at least one correct answer appears among the first one or ten ranked entities
- nDCG@10 additionally rewards correct answers placed near the top and accounts for multiple correct answers

**Project-relevant formulas:**

$$
\operatorname{RetrievalGoldCoverage}=
\frac{1}{|Q|}\sum_{q\in Q}\frac{|G_q\cap R_q|}{|G_q|}
$$

$$
\operatorname{RetrievalFullGoldCoverage}=
\frac{1}{|Q|}\sum_{q\in Q}\mathbb I(G_q\subseteq R_q)
$$

**Example:** If a question has three correct answers and two are retrieved, its gold coverage is $2/3$, but it does not count as fully covered.

## Part III: Experimental results

### Slide 13: Резултати од пребарувањето

**Target time:** 1:20

**Purpose:** Answer which architecture retrieves most effectively.

**Include:**

- Architecture comparison chart with mean and standard deviation across three seeds
- NBFNet: Hits@1 of 0.692
- NBFNet: Hits@10 of 0.921
- NBFNet: nDCG@10 of 0.796
- NBFNet: RetrievalGoldCoverage of 0.877
- NBFNet: RetrievalFullGoldCoverage of 0.827

**Discuss:**

- NBFNet is strongest on every principal retrieval metric and has low variability
- ReaRev is the second strongest architecture but is less stable
- Advance GraphSAGE provides a strong improvement over the baseline while remaining comparatively simple and fast to train

**Main message:** Direct question conditioning and explicit relational-path modeling are particularly effective for WebQSP retrieval.

### Slide 14: Конструкција на доказниот подграф

**Target time:** 1:00

**Purpose:** Introduce the two methods that transform retrieved candidates into evidence.

**Include:**

- Union of all shortest paths from the seed entities to each reachable candidate
- PCST selection of a compact rooted connected structure
- Rank-based candidate prizes
- Constant and semantic edge costs

**Essential objective:**

$$
T^*=\arg\max_T\left[
\sum_{v\in V_T}p(v)-\sum_{e\in E_T}c(e)
\right]
$$

**Cost variants:**

$$
c(e)=\lambda
$$

$$
c(e)=\max\!\left(\varepsilon,\lambda\left(1-\cos(q,r_e)\right)\right)
$$

**Visual:** Apply both methods to the same small candidate graph.

### Slide 15: Метрики за доказниот контекст

**Target time:** 0:45

**Purpose:** Explain how compactness and retained information are evaluated together.

**Explain without formulas:**

- AverageTriples describes context size
- ContextCandidateCoverage measures the proportion of retrieved candidates preserved in the context
- CandidateReduction measures the total percentage of retrieved candidates removed

**Project-relevant formulas:**

$$
\operatorname{ContextGoldCoverage}=
\frac{1}{|Q|}\sum_{q\in Q}\frac{|G_q\cap C_q|}{|G_q|}
$$

$$
\operatorname{ContextFullGoldCoverage}=
\frac{1}{|Q|}\sum_{q\in Q}\mathbb I(G_q\subseteq C_q)
$$

**Main message:** A smaller evidence subgraph is useful only if it preserves the entities and relations required for answering the question.

### Slide 16: Компактност и зачувување на точните одговори

**Target time:** 1:20

**Purpose:** Present the evidence-subgraph experiment.

**Include:**

- Shortest paths: 94.47 average triples
- PCST: approximately 13.5 to 14.5 average triples
- Approximately 85% context reduction
- ContextGoldCoverage decreases from 0.888 to approximately 0.881 for most PCST configurations
- ContextFullGoldCoverage decreases from 0.838 to approximately 0.830 for most PCST configurations
- Semantic PCST with $\lambda=1$ retains 0.991 of the candidates

**Discuss:**

- The effect of changing $\lambda$ is limited across most of the tested interval
- The cost strategy matters more than small changes in $\lambda$
- Semantic costs preserve candidates more effectively than constant costs at $\lambda=1$

**Main message:** PCST removes most triples while preserving nearly the same entity-level gold coverage.

### Slide 17: Метрики за конечниот одговор

**Target time:** 0:50

**Purpose:** Explain the final-answer metrics and isolate LLM omissions.

**Explain without formulas:**

- Hit indicates whether the LLM returned at least one correct answer
- ExactMatch requires the predicted and gold answer sets to match completely
- Precision, Recall and F1 evaluate answer-set quality across all questions
- FullContextCompleteAnswer counts questions where the context contains every gold answer and the LLM returns all of them
- FullContextLLMOmission counts questions where the context contains every gold answer but the LLM omits at least one

**Conditional metric:**

$$
\operatorname{LLMOmissionGivenFullContext}=
\frac{\#\{q:G_q\subseteq C_q\ \land\ G_q\nsubseteq P_q\}}
{\#\{q:G_q\subseteq C_q\}}
$$

**Main message:** This conditional metric measures LLM failure only among questions for which the complete gold answer set was available in the evidence context.

### Slide 18: Квалитет и потрошувачка на токени

**Target time:** 1:40

**Purpose:** Present the end-to-end comparison and its practical trade-off.

**Include:**

- DeepSeek-V4-Flash with shortest paths: Hit 0.802 and F1 0.627
- GPT-5.6 Luna with shortest paths: Hit 0.804 and F1 0.613
- Semantic PCST with $\lambda=1$: Hit 0.689 for DeepSeek and 0.728 for GPT
- Shortest paths: approximately 3.07 to 3.42 million tokens
- PCST: approximately 0.66 to 0.72 million tokens
- PCST reduces token use by approximately 78% to 80%

**Discuss:**

- Shortest paths provide the highest final quality
- Semantic PCST performs better than constant PCST at a similar context size
- GPT achieves the highest Hit, while DeepSeek achieves the strongest F1 with shortest paths
- Semantic PCST with $\lambda=1$ and GPT uses approximately 4.6 times fewer tokens than shortest paths while reaching a Hit of 0.728

**Visual:** Use separate quality and token-consumption plots or a clear quality-versus-token trade-off plot.

### Slide 19: Каде се губи информацијата?

**Target time:** 1:40

**Purpose:** Present the central analytical finding of the thesis.

**Include:**

- NBFNet RetrievalGoldCoverage: 0.877
- Shortest-path ContextGoldCoverage: 0.888
- Shortest-path ContextFullGoldCoverage: 0.838
- Shortest-path LLMOmissionGivenFullContext: 0.211 for DeepSeek and 0.237 for GPT
- Semantic PCST with $\lambda=1$ preserves ContextFullGoldCoverage of approximately 0.830
- The same semantic PCST configuration has LLMOmissionGivenFullContext of approximately 0.477 for both LLMs

**Important explanation:** ContextGoldCoverage may be slightly higher than RetrievalGoldCoverage because evidence construction can include a gold entity as an intermediate node even when the retriever did not select it directly.

**Main message:** Preserving the correct entity is insufficient. The LLM also needs relational triples that make the connection between the question and the answer clear.

## Part IV: Answers and conclusion

### Slide 20: Одговори на истражувачките прашања

**Target time:** 1:30

**Purpose:** Return to the three questions from Slide 5 and answer each directly.

**Answer 1: Architecture choice**

NBFNet provides the strongest and most stable retrieval because it explicitly models question-conditioned relational paths from the seed entities to candidate answers.

**Answer 2: Evidence construction**

The union of shortest paths preserves the evidence required for the strongest final answers. Semantic PCST provides the best compromise when compact context and lower token consumption are priorities. The cost strategy matters more than small changes in $\lambda$.

**Answer 3: Language-model choice and context utilization**

The two language models perform similarly with shortest-path evidence. GPT-5.6 Luna achieves the highest Hit, while DeepSeek-V4-Flash achieves the strongest F1. The principal limitation appears in the transition from available evidence to the generated answer, especially after PCST compression.

**Final conclusion:**

> Успешното одговарање не зависи само од пронаоѓањето на точниот ентитет, туку и од зачувувањето и искористувањето на релациите што објаснуваат зошто тој ентитет е точниот одговор.

**Future direction:** Briefly mention a hybrid evidence strategy that retains PCST compactness while guaranteeing complete and interpretable relational paths for the most important candidates.

### Slide 21: Ви благодарам

**Target time:** 0:15

**Purpose:** End the timed presentation and invite questions.

**Include:**

- „Ви благодарам за вниманието“
- „Прашања?“
- Candidate name
- Optional repository link or QR code

**Visual:** Keep the slide minimal and visually consistent with the title slide.

## Timing summary

| Part | Slides | Target time |
|---|---:|---:|
| Problem and research questions | 1–5 | 3:20 |
| Methodology | 6–12 | 4:30 |
| Experimental results | 13–19 | 8:35 |
| Answers and closing | 20–21 | 1:45 |
| **Total** | **21** | **18:10** |

The remaining 1 minute and 50 seconds provide a safety margin for transitions, pauses and small deviations during delivery.

## Formula policy

- Use formulas only when they clarify the defining mechanism of an architecture or a project-specific metric.
- Keep at most one or two formulas on an architecture slide.
- Explain the architectural intuition rather than deriving formulas term by term.
- Explain familiar metrics such as Hits, Precision, Recall and F1 verbally.
- Show formulas for project-specific coverage and omission metrics because their denominators affect interpretation.
- Show only the PCST objective and the two edge-cost definitions.

## Visual direction

- Use a 16:9 layout.
- Keep slide titles in Macedonian and without trailing periods.
- Use blue for input, orange for retrieval, green for evidence construction and purple for answer generation.
- Reuse the visual language of the thesis architecture figures, but adapt complex figures for projection and presentation distance.
- Avoid screenshots of thesis pages.
- Prefer one principal visual or chart per slide.
- Label important values directly on result charts where possible.
- Keep equations large and explain only the terms needed for the slide's main point.
- Use short slide text. The spoken presentation should carry the detailed explanation.

## Backup slides

The following slides should appear after Slide 21 and should not count toward the timed presentation:

1. Full architecture comparison table
2. Complete architecture characteristics table
3. Full PCST $\lambda$ sensitivity results
4. Complete evidence-subgraph results table
5. Complete end-to-end results table
6. Exact definitions and denominators of all metrics
7. Training configuration and filtering rules
8. Limitations and architecture adaptations

