---
date: 2026-03-05
categories:
  - 學習筆記
tags:
  - Reingforcement Learning
---


# Markov Decision Process

## Table of Contents
- [RL基本想法](#rl-basics)
- [Markov Process](#markov-process)
- [Value Function, Q-function, and Policy](#value-q-policy)
- [from State Value Function to Bellman Equation](#state-value-to-bellman)

## RL基本想法 { #rl-basics }
### RL的組成與互動

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


### RL的學習目標

- 透過Reward的反饋與累積，學習最佳的*Policy*(如何決定Action)。
- 目標: 最大化*Expected Cumulative Reward* (期望累積獎勵) 。
- 假定環境的未來狀態有可預期的結束，則稱為 *Episodic Task*；反之則為*Continuing Task*。
- 為了解決在Continuing Task中無法定義終止狀態的問題，引入了*Discount Factor* (折扣因子) $\gamma$，用於折扣未來獎勵的價值，使得總獎勵有限。

> 可定義 $G_t (在t時刻的累積獎勵)為:$
> $$G_t = \sum_{i=t}^{\infty} \gamma^{i-t} R_i \ , \gamma \in [0,1]$$


* 當$\gamma$ 接近 0 時: 近期的結果較為重要(*短視近利*)
* 當$\gamma$ 接近 1 時: 未來的結果較為重要(*高瞻遠祖*)

## Markov Process { #markov-process }

### Markov property

- 未來只取決於現在， **與過去如何走到現在無關**
- Aka *Memoryless*
- 也就是說 $S_{t+1}只被S_t決定，其變成S之機率P表示: P(S_{t+1}|S_t)$
- 如果把*History* 歷史資訊考慮進來，則也可表示為 $P(S_{t+1}|S_0, S_1 ... S_{t-1},S_t)$



### Markov Family

1. Markov process(MP): 
    - 綜合考量State與其*state transition probability*(狀態轉換機率)，則Markov process 即為 $\{S,P\}$
    - > $P蘊含所有state 的轉換機率: \forall s,s' \in S, P(s'|s) = P(S_{t+1} = s' | S_t = s)$

2. Markov reward process(MRP):
    - 將Reward考慮至MP中，其中R為一所有State 至其 Reward的對映
    - MRP = $\{S,P,R\}$


## Value Function, Q-function, and Policy { #value-q-policy }

### Value Function

- 在MDP中，我們如何考慮一個State s有多好?
- *Ans:* 定義一 *value function $V$*未來狀態的Reward 期望值總和

$$V(s) = \mathbb{E}[G_t|s_t = s] \newline G_t = \sum_{i=t}^\infty \gamma^{i-t}r_i$$

- 透過一系列的分解，我們可以將value function 降解為 $V(s) = \mathbb{E}[r_t + \gamma V(S_{t+1})|s_t = s]\leftarrowtail 又稱 \textbf{Bellman Equation}$
- 相當於: 目前的獎勵與未來的獎勵
- Value function 可以透過以下方式推導(學習):
    1. Dynamic Programming
    2. Monte-Carlo
    3. TD Learning 

### MDP- Markov Decision Process

- 在RL裡，action 會影響state 的狀態轉變，因此transition probability要考慮action。
>  $p_{ss'}^a = P(s_{t+1}= s' | s_t = s, a_t = a)$

- MDP = MRP + 考慮Action = $\{S,P,R,A\}$

### Policy

- 決定agent如何在環境下做選擇的機率分布
> $\pi(a|s) = P[a_t = a | s_t = s]$

- Policy包含:
    1. 決定可行動作有甚麼
    2. 決定動作時機
    3. 決定實際動作的挑選

- Policy 會影響value fuction 如何評估state
> $V_{\pi}(s) = \mathbb{E}_{\pi}[G_t | s_t = s]$

### Q-function

- 衡量當前state採取action後在當前policy下*expected return*的函式稱為 *Q-function*
> $Q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | s_t = s, a_t = a]$

- 比較 $V_{\pi}(s) 與 Q_{\pi}(s,a)$:

| $\pi$ | 策略- agent 行事風格 | E.g. 駕駛習慣 |
| ------|--------------------|--------------|
| $Q_{\pi}(s,a)$| 當前行動對於策略有多好 | E.g. 路口決定左轉有多好|
| $V_{\pi}(s)$ |  當前狀態有多好    | E.g. 當前位置有多好 |

## from State Value Function to Bellman Equation { #state-value-to-bellman }

$$
\begin{aligned}
V_{\pi}(s)
&= \mathbb{E}_{\pi}[G_t \mid s_t = s]
&& \color{#1e40af}{\text{[Step 1] Basic definition}} \\
&= \mathbb{E}_{\pi}[r_t + \gamma G_{t+1} \mid s_t = s]
&& \color{#1e40af}{\text{[Step 2] Expand return}} \\
&= \sum_{a}\pi(a \mid s)\sum_{s'}p(s' \mid s,a)\left[r_t + \gamma\,\mathbb{E}[G_{t+1} \mid s_{t+1}=s']\right]
&& \color{#1e40af}{\text{[Step 3]  Expand using policy and transitions}} \\
&= \sum_{a}\pi(a \mid s)\sum_{s'}p(s' \mid s,a)\left[r_t + \gamma V_{\pi}(s')\right]
&& \color{#1e40af}{\text{[Step 4] Recursive form (Bellman equation)}}
\end{aligned}
$$

### Find the Best

1.  Best value possible from any strategy: 
    - $\text{Optimal State Value Function: } V_*(s) = \mathop{max}\limits_{\pi}V_{\pi}(s)$
2. Best action-value from any strategy
    - $\text{Optimal Q Function: } Q_*(s,a) = \mathop{max}\limits_{\pi} Q_{\pi}(s,a)$ 

## Some Variation

### Partially Observable MDP (POMDP)
* Environment 不再完全可被觀測時，需要有額外的:
    1. $\text{Observation: 對於當前環境的認知、觀測} \:O$ 
    2. $\text{Observation function} \:Z_{s'o}^a = P(o_{t+1} = o | s_{t+1} = s', a_t = a)$ 
* 此時可定義 POMDP 為 MDP + 觀測與觀測函式: $\{S, P, R, A, O, Z\}$

### Uncertain Environment
* 有時Next State 完全無法得知

### Deterministic Policy vs. Stochastic Policy

- **Deterministic**: When One State Always Leads to One Action
  - $a = \pi(s)$
- **Stochastic**: A policy outputs a probability distribution over actions
  - $a \text{~}\pi(a|s) = \mathbb{P}[a_t = a|s_t = s]$

> 通常要選擇使用 Stochastic Policy- Explores options + Adapts naturally + Keeps them
guessing

## Q&A

1. 為甚麼不把 $\gamma$ 設很大($\lesssim 1$)就好?