| Metric | Simple meaning | How it is calculated | Use |
|---|---|---|---|
| Hit@3 | Did the correct chunk appear in the first 3 results? | Questions with a correct chunk in top 3 ÷ total questions | Measures whether retrieval finds the answer quickly |
| MRR | How high was the first correct result? | `1/rank` — rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33 | Rewards correct chunks appearing near the top |
| Precision | How many retrieved chunks were actually relevant? | Relevant retrieved chunks ÷ all retrieved chunks | Detects noisy or irrelevant retrieval |
| Recall | How many relevant chunks were found? | Relevant retrieved chunks ÷ all expected relevant chunks | Detects missed evidence |
| F1 | Balance between precision and recall | `2 × precision × recall ÷ (precision + recall)` | Gives one combined retrieval score |
| Faithfulness | Is the answer supported by retrieved policy text? | Validator checks the answer against retrieved evidence | Detects hallucinations |
| Answer relevancy | Does the answer address the question? | Current script compares question words with answer words | Detects answers that go off-topic |
| Answer correctness | Does the answer match the expected answer? | Current script compares ground-truth words with answer words | Measures factual agreement |