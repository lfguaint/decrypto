# Decrypto LLM Benchmark — Contexto do Projeto

## Visão Geral

Benchmark que simula o jogo de tabuleiro **Decrypto** com três agentes LLM para avaliar capacidades distintas de linguagem:

- **Encryptor**: produz pistas sem deixar rastros exploráveis
- **Decoder**: recupera o código a partir das pistas, conhecendo as keywords
- **Interceptor**: deduz o código apenas das pistas e do histórico público, sem conhecer as keywords

## Mecânica do Jogo

- 4 keyword cards secretas (conhecidas por Encryptor e Decoder; desconhecidas pelo Interceptor)
- Cada rodada: Encryptor recebe um código de 3 dígitos distintos de {1,2,3,4} e dá 3 pistas de uma palavra
- Interceptor chuta primeiro; Decoder chuta independentemente (sem ver o guess do Interceptor)
- Ao final de cada rodada: código real revelado a todos os três jogadores
- Histórico público cresce a cada rodada (pistas, guesses, códigos reais)
- Códigos sorteados com reposição (o mesmo código pode sair mais de uma vez no jogo), como no manual oficial
- Jogo de 5 rodadas

## Classificação de Resultados por Rodada (S / I / F)

- **S** (success): decoder acerta E interceptor erra
- **I** (intercepted): interceptor acerta — independente do decoder
- **F** (failure): decoder erra E interceptor erra
- Quarto estado (decoder erra + interceptor acerta) → classificado como **I**

## Decisões Metodológicas

### Verificação independente
Decoder e interceptor não veem o guess um do outro na rodada corrente. Permite medir capacidade bruta de cada papel de forma separável. Diverge do jogo real (onde o decoder vê o resultado do interceptor antes de responder) — documentar como escolha metodológica no artigo.

### Decoder sempre joga
Mesmo após interceptação o decoder responde. Dado analítico extra: separa "pista ruim" (ambos erram) de "interceptor esperto" (decoder teria acertado).

### Estatística
Wilson 95% CI em todas as proporções.

## Design do Experimento Principal

- **Fixed-background**: 3 séries (varia Encryptor / Decoder / Interceptor), dois papéis fixos no modelo de referência DeepSeek V4 Flash
- **10 modelos × 3 séries = 28 configurações únicas** (a configuração de referência, com os três papéis no DeepSeek Flash, aparece nas três séries e conta uma vez) **× 500 keyword sets × 1 jogo × 5 rodadas = 14.000 jogos**
- **Temperatura**: não enviada (default de cada provedor) — reasoning models ignoram o parâmetro
- **Prompt**: só a v3, para os três agentes
- **Sementes**: benchmark = **3** (as 28 configurações e os sub-estudos, todos pareados); pilotos e testes = **42** (nunca usar a 3 para ajustar código ou prompts). Os códigos usam `seed` (fluxos `"3-codes-k-g"`, um por jogo) e as keywords usam `keyword_seed` (fluxo `"3-keywords"`; padrão = `seed`, e também 3 no benchmark)
- **Keywords**: lista de 414 palavras congelada por hash SHA-256 em `keywords.py`; os 500 sets canônicos (ids 1–500) estão em `data/keyword_sets_seed3.json`, e o benchmark os lê de lá com `--keyword-sets-file data/keyword_sets_seed3.json`
- Custo: a estimativa antiga (~$75 base / ~$460 com reasoning) era para 15 modelos e precisa ser refeita

### Painel (10 modelos, lista canônica em `config.py`)

- **T1:** Gemma 4 31B, gpt-oss-120b, Qwen3 32B
- **T2:** DeepSeek V4 Flash (referência), Gemini 2.5 Flash, GPT-6 Luna, Claude Haiku 4.5
- **T3:** DeepSeek V4 Pro, Kimi K3, GPT-6.1 Sol

Pares de comparação candidatos (a decisão de parear ou não fica para depois dos resultados; esta lista foi registrada antes deles):

| Tipo | Par | Ressalva |
|---|---|---|
| Escala na família | DeepSeek V4 Flash × V4 Pro | mesma geração: par limpo |
| Escala na família | GPT-6 Luna × GPT-6.1 Sol | versões diferentes (6 × 6.1): escala e versão se confundem |
| Aberto × fechado | Gemma 4 31B × Gemini 2.5 Flash | gerações diferentes |
| Aberto × fechado | gpt-oss-120b × GPT-6 Luna | Luna por ser a fechada mais barata (espelha T1 × T2 do par Google); Sol também serviria |

A Anthropic só tem o Claude Haiku 4.5 no painel, então não há par de escala da Anthropic.

## Prompt v3 (Final)

Congelado em 2026-10-26: `prompts.PROMPTS_SHA256` (o import falha se algum texto mudar), hash gravado em cada experimento, e cópia canônica exata em `data/prompts_v3.txt`.

Princípio: **descrever as regras do jogo fielmente, sem prescrever estratégia**. Permite distinguir modelos que entendem o jogo de quem não entende.

### Encryptor
```
You are the Encryptor in a word-clue game with three players: you (Encryptor), a Decoder teammate, and an Interceptor opponent.

Your team's 4 secret keyword cards, known to you and the Decoder:
  1: <KEYWORD1>
  2: <KEYWORD2>
  3: <KEYWORD3>
  4: <KEYWORD4>

Each round you receive a 3-digit code — a sequence of 3 digits from {1, 2, 3, 4}, each appearing exactly once.

**Your goal:**
- Encode the 3-digit code as 3 one-word clues — the first clue corresponds to the first digit, the second to the second, and the third to the third.
- Your aim: the Decoder (who knows the keywords) recovers the code from your clues; the Interceptor (who does not know the keywords) does not.

**How the game works:**
- After your clues, the Interceptor submits their guess first, then the Decoder submits theirs independently — neither sees the other's guess for the current round.
- If the Interceptor guesses correctly, the round is intercepted regardless of the Decoder's guess.
- At the end of each round, the actual code is revealed to all three players.
- Each round's code is drawn uniformly at random from all possible codes, independently of earlier rounds, so the same code may come up more than once.
- The public record grows each round: after a round ends, its clues, both guesses, and the actual code remain visible to all three players in every subsequent round.

**Rules for clues:**
- Each clue must be a single word with no digits.
- No keyword may be used as a clue, not even for a different keyword.
- A clue must relate to the meaning of the keyword it encodes, not to its spelling, letters, sound or position.
- No clue word may be used more than once, even across different rounds.
```

### Decoder
```
You are the Decoder in a word-clue game with three players: an Encryptor teammate, you (Decoder), and an Interceptor opponent.

Your team's 4 secret keyword cards (unknown to the Interceptor):
  1: <KEYWORD1> / 2: <KEYWORD2> / 3: <KEYWORD3> / 4: <KEYWORD4>

Each round your Encryptor gives 3 one-word clues in order — the first clue corresponds to the first digit of the code, the second to the second, and the third to the third.

**Your goal:**
- Recover the 3-digit code: a sequence of 3 digits from {1, 2, 3, 4}, each appearing exactly once.

**How the game works:**
- The Interceptor submits their guess before you; you submit yours independently — you do not see the Interceptor's guess for the current round.
- At the end of each round, the actual code is revealed to all three players.
- Each round's code is drawn uniformly at random from all possible codes, independently of earlier rounds, so the same code may come up more than once.
- The public record grows each round: after a round ends, its clues, both guesses, and the actual code remain visible to all three players in every subsequent round.
```

### Interceptor
```
You are the Interceptor in a word-clue game with three players: an Encryptor, a Decoder, and you (Interceptor).

The Encryptor and Decoder share 4 secret keyword cards numbered 1 to 4. You do not know what the keywords are.

Each round the Encryptor gives 3 one-word clues in order — the first clue corresponds to the first digit of the code, the second to the second, and the third to the third.

**Your goal:**
- Guess the 3-digit code: a sequence of 3 digits from {1, 2, 3, 4}, each appearing exactly once.

**How the game works:**
- You submit your guess before the Decoder; neither of you sees the other's guess for the current round.
- If you guess correctly, the round is intercepted regardless of the Decoder's guess.
- At the end of each round, the actual code is revealed to all three players.
- Each round's code is drawn uniformly at random from all possible codes, independently of earlier rounds, so the same code may come up more than once.
- The public record grows each round: after a round ends, its clues, both guesses, and the actual code remain visible to all three players in every subsequent round.
```

## Resultados Preliminares (Calibração)

### Gemma 3 27B self-play (500 jogos, prompt v2)
- S ≈ 47%, I ≈ 28%, F ≈ 25%
- ~40s/jogo em média
- Prior rigidity: 500/500 jogos com guess R1 = [1,2,3] (modelo colapsa ao prior)

### Nemotron 70B como Interceptor (12 jogos, prompt v2)
- I = 63.3% [50.7–74.4%] — muito superior ao Gemma como interceptor
- Curva por rodada: 0% → 50% → 92% → 75% → 100%
- ~100–160s/rodada (reasoning traces longas); ~627s/jogo
- Prior R1: 11/12 jogos com [1,2,3]

## Framework Bayesiano (para o artigo)

- **Interceptor**: estima P(cₜ | hₜ, H_{t-1}) marginalizando sobre θ (mapeamento keyword)
- **Decoder**: tem P(cₜ | hₜ, θ, H_{t-1}) com θ conhecido
- **Encryptor**: otimiza objetivo dual — maximizar P(decoder acerta) e minimizar P(interceptor acerta)
- Cada pista atualiza permanentemente P(θ | H) — vazamento de informação irreversível
- 24 códigos possíveis (permutações de 3 em {1,2,3,4}), sorteados com reposição (chance de acerto ao acaso sempre 1/24)

## Próximos Passos

- [ ] Rodar benchmark principal (10 modelos, 500 sets, OpenRouter)
- [ ] Implementar `run_grid.py` para 28 configurações
- [ ] Escrever artigo

## Arquivos Principais

- `prompts.py` — prompts v3 dos três agentes (v1/v2 removidos; ficam no histórico do git)
- `models.py` — estruturas de dados (GameRecord, RoundRecord, métricas)
- `agents.py` — classes Encryptor, Decoder, Interceptor
- `runner.py` — runner paralelo
- `storage.py` — SQLite storage
- `config.py` — RunConfig, AgentConfig, lista de modelos
- `results/` — bancos SQLite dos experimentos
