---
date: 2026-03-05
categories:
  - 學習筆記
tags:
  - Machine Learing
---

# Deep Reinforcement Learning

## Summary

|Lecture|Description|Time|
|-------|-----------|----|
|1      | [Markov Decision Process](#Markov-decision-process)  |0302|


## Markov Decision Process

### RL基本想法
#### RL的組成與互動

- RL的組成:
  -  Agent: 代理者，在環境中做出行動的實體。 E.g. 自動駕駛車輛、機器人等。
  -  Environment: 環境，代理者所處的外部世界，包含所有可能的狀態和動作。 E.g. 道路、交通狀況等。

> RL 是代理者在環境的互動中學習如何做出最佳決策的過程。

- Agent 與 Environment 的互動:
  1. Agent 根據當前的*State* 選擇一個 *Action*。
  2. Environment 根據 Agent 的 Action **更新** State，並給予 Agent 一個 *Reward*。
  3. Reward 可能會影響 Agent 未來的行動選擇。

> Suppose:  
> $S_t \in \mathcal{S}(State Space)$; 
> $A_t \in \mathcal{A}(Action Space)$; 
> $R_t \in \mathcal{R}(Reward Function)$;
> The interaction be like:   
> $S_t,R_t \rightarrow \fbox{Agent} \rightarrow A_t \rightarrow \fbox{Emvironment} \rightarrow S_{t+1},R_{t+1}$  


#### RL的學習目標

- 透過Reward的反饋與累積，學習最佳的*Policy*(如何決定Action)。
- 目標: 最大化*Expected Cumulative Reward* (期望累積獎勵) 。
- 假定環境的未來狀態有可預期的結束，則稱為 *Episodic Task*；反之則為*Continuing Task*。
- 為了解決在Continuing Task中無法定義終止狀態的問題，引入了*Discount Factor* (折扣因子) $\gamma$，用於折扣未來獎勵的價值，使得總獎勵有限。

> 可定義 $G_t (在t時刻的累積獎勵)為:$
> $$G_t = \sum_{i=t}^{\infty} \gamma^{i-t} R_i \ , \gamma \in [0,1]$$


* 當$\gamma$ 接近 0 時: 近期的結果較為重要(*短視近利*)
* 當$\gamma$ 接近 1 時: 未來的結果較為重要(*高瞻遠祖*)

### Markov Process

#### Markov property

- 未來只取決於現在， **與過去如何走到現在無關**
- Aka *Memoryless*
- 也就是說 $S_{t+1}只被S_t決定，其變成S之機率P表示: P(S_{t+1}|S_t)$
- 如果把*History* 歷史資訊考慮進來，則也可表示為 $P(S_{t+1}|S_0, S_1 ... S_{t-1},S_t)$



#### Markov Family

1. Markov process(MP): 
  - 綜合考量State與其*state transition probability*(狀態轉換機率)，則Markov process 即為 $\{S,P\}$
  - > $P蘊含所有state 的轉換機率: \forall s,s' \in S, P(s'|s) = P(S_{t+1} = s' | S_t = s)$

2. Markov reward process(MRP):
  - 將Reward考慮至MP中，其中R為一所有State 至其 Reward的對映
  - MRP = $\{S,P,R\}$
### Q&A

1. 為甚麼不把 $\gamma$ 設很大($\lesssim 1$)就好?

### Bellman Equation

#### Value Function
