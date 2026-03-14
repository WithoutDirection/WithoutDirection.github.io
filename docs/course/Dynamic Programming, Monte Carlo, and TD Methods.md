# Dynamic Programming, Monte Carlo, and TD Methods

## Table of Contents

## Find the best policy in MDP

### Recap: what is MDP
- 一言以蔽之: state的轉換機率只與當前state和action有關，與過去的state和action無關。
- MDP = {S(state), P(transition probability), R(reward), A}
- 在MDP中，目標是找到最好的policy $\pi^*$，使得在該policy下的value function $V_{\pi}(s)$最大化。
- 通常利用求解value function(*Bellman Optimality Equation*)找到每個狀態的最優價值函數 $V^*(s)$，進而推導出最佳策略

> Bellman Equation: $V(s) = \sum\limits_{s',r} p(s',r | s, a)[r + \gamma V(s')]$  
> 現在的價值 = 動作(a)後立刻得到的好處 + 折扣過後($\gamma$)未來的好處(V(s'))


### Know the whole model vs. Istead
- MDP完全可知 = 知道所有*State* + 知道 state之間的*轉換機率*  + State的轉換*Reward* + 可行的*動作集合*
  
- 當MDP為可知時(通常針對**P- 狀態之間的轉換機率**)，可以透過**Dynamic Programming** 求得最好的value，進而求得最好的policy
- 反之，當MDP不完全可知時，之能透過*Model free function***學習**出最佳策略

## Use Dynamic Programming solve MDP

### DP要素
- Value function(i.e. *Bellman Equation*)滿足以下條件

1. Optimal Substructure
   - 在MDP中，最佳策略的子策略也是最佳的。
   - 也就是說，如果一個策略在某個狀態下是最佳的，那麼在該狀態下採取的行動也是最佳的。
2. Overlapping Subproblems
   - 在MDP中，不同的狀態之間可能存在重疊的子問題，這使得動態規劃方法能夠有效地解決。
- DP方法的核心思想是利用這些特性，通過遞歸地求解子問題來找到整體問題的最佳解。

### DP步驟

1. Decomposition by Bellman equation
    - 每個狀態的最佳價值取決於其未來的最佳價值
2. Computed state values can be stored and reused
    - 將計算而得的state value存起來讓計算在其之前的狀態時可以直接使用

### DP如何求得Best Policy

1. Policy Evaluation
   1. 衡量當前policy的價值: 計算出當前策略在每個狀態下的預期回報
   2. **遞迴計算直到數值收斂**
   3. 建立一張完整的價值地圖（value map）-當前policy的優劣
2. Policy Improvement
   1. 因為*policy evaluation*，可以知道當前策略最好的下一步是甚麼
   2. 採取*Greedy principle*，在每一個*state*下根據第一步得到的**value mao**，直接選擇能通往「下一步最高價值」的action來取代舊有行為
   * 因為*greedy principle*，其保證直接選擇最大value的action一定比目前的策略好或是至少一樣好 
- 根據上述**1.**, **2.**交替進行evlaute與imporve(greedy choice)，最後求得的value必為最大值
> Procedure: $\pi_0 \xRightarrow{evaluate} V_{\pi0} \xRightarrow{imporve} \pi_1 \xRightarrow{evaluate} V_{\pi1} \text{\dots} \xRightarrow{imporve} \pi^* \xRightarrow{evaluate} V^*$ 

> Recap: 用Bellman equation衡量policy價值  
> $V_{k+1}(s) \leftarrow \mathbb{E}[r_t + \gamma V_k(s_{t+1})| s_t = s] = \sum\limits_a \pi(a|s) \sum\limits_{s'}p(s'|s,a)[r(s,a,s') + \gamma V_k(s')]$
> That the value of being in state is: 
> 1. 立刻得到的獎勵 - $r(s,a,s')$
> 2. 折扣後的未來獎勵 $\gamma V_k(s')$
> 3. 給定狀態 $s$ 下所有可能行動 $a$  - $\sum_a \pi(a|s)$
> 4. 所有可能的未來狀態 - $\sum_{s'} p(s'|s, a)$


## Model Free Method 
> Recap: *Model Free* = 不知道完整的模型樣貌。 E.g. 不知道 S中所有的state

- Learn from *experience*: sequence of $(s_t, a_t, r_t, s_{t+1})$
- 因為無法得知完整的模型，不再能夠用**Bellman Equation**衡量*state*或是 *policy*的 value
- 需要從經驗中學習:
   1. Model-Free Prediction: 如何在未知中評估現狀
   2. Model-Free Control: 如何在未知中尋找最佳策略

### Model-Free prediction
* Understanding the environment's patterns
   * Estimate value functions of unknown MDPs through direct experience
   * Answer the question: ```How good is this situation if I follow my current strategy?```
   * Approaches: *Monte Carlo (MC) methods*, *Temporal-Difference (TD) learning*
   * **Mathematical goal:** $\text{Approximate} V_{\pi}(s) \text{or } Q_{\pi}(s,a) \text{without knowing } p(s'|s,a)$

### Model-Free Control
* Mastering optimal behavior
   * Discover optimal policies without transition or reward models
   * Address the challenge: ```What's the best possible strategy in this unknown environment?```
   * Approached: *Q-learning*, *SARSA*, and *policy gradient methods*
   * Mathematical goal: $\text{Find }\pi^*(s) = \arg\max Q^*(s,a)\text{ through interactions}$

## Model-Free prediction

### Monte Carlo (MC) Prediction
- Episodic Learning Focus: 要有**開始**與**結束**的明確定義
- Experience-Driven Valuation: 評估函數為將觀察到的**return value** 平均而得
   - i.e. 完全不依賴自己推測、預估，完全從經驗推得
- Complete Trajectory Sampling: 收集**完整的**互動回合並將其作為該策略的*return*
   - $\text{Return Calculation} = G_t = \sum\limits_{i=t}^T \gamma^{i-t}R_i \\ \text{T is the terminal step of an }\bold{episode}$


#### Algo Detail of First-Visit Monte Carlo
```c
Algorithm: Monte Carlo Prediction 

Require: Policy π to be evaluated   // 待評估的策略 $\pi$ (Policy)
Ensure: Value function V ≈ v_π   // 價值函數 V:該策略下各狀態的估計價值

// Step 1. 初始化價值：對所有狀態s設定V(s)為任意值（通常設為 0）。
Initialize V(s) arbitrarily for all s ∈ S {Often set to zero} 
// Step 2. 建立紀錄表：初始化一個空列表Returns(s)，用來存放每個狀態以後觀察到的所有回報。
Initialize Returns(s) ← empty list, for all s ∈ S {Track all returns observed}

// Step 3. 根據經驗逐漸逼近
repeat
   Generate an episode following π: //Step 3.1 根據策略產生一個完整的回合 (Episode)：
   s0, a0, r1, s1, a1, ..., rT {Complete trajectory}
   G ← 0 {Initialize return}
   // Step 3.2 從回合最後一刻倒著算回去 t = T-1, T-2, ..., 0)
   for t = T−1, T−2, ..., 0 do 
      // 逐漸更新總回報：G <- γG + r_{t+1}（考慮折扣因子 γ）
      G ← γG + r_{t+1} {Update return with discount} 
      // 如果狀態 $s_t$ 在此時間點之前: 將本次算出的 $G$ 加入到 $Returns(s_t)$ 列表中。並且計算平均值：V(s_t) ← average(Returns(s_t))。
      if s_t ∉ {s0, s1, ..., s_{t−1}} then
         Append G to Returns(s_t) {Store return for this state}
         V(s_t) ← average(Returns(s_t)) {Update value estimate}
      end if
   end for
until sufficient sampling

```

- 為什麼要檢查「第一次遇到」 (Step 8)？
   - 這稱為 First-visit MC。
   - 如果在一個回合內兩次經過同一個格子（例如：繞了一圈），只記錄第一次進入該格子的總分數。
   - **這樣可以確保數據的統計特性更純粹，避免同一個事件被重複加權。**

#### Incremental Monte Carlo Update
> Traditional MC: 完成很多次回合後再算平均

- Incremental Monte Carlo Update: 每完成一個回合，就立刻**根據新看到的分數來「微調」目前的價值估計**，
   Advantage: 不需要儲存龐大的分數列表。
- 對於一個狀態 $s$，當我們完成一個回合並得到回報 $G_t$ 時，更新公式如下: $V(s) \leftarrow V(s) + \alpha (G_t - V(s))$
   - $V(s)$：舊的價值估計
   - $G_t$：新看到的回報
   - $\alpha$：學習率（步長）
   - $(G_t - V(s))$：預測誤差 (Estimation Error)

> **公式推導**: $\\\text{Traditional MC} = \mu_k \\= \frac{1}{k}\sum\limits_{i = 1}^k x_i \\= \frac{1}{k}(x_k + \sum\limits_{i = 1}^{k - 1}) \\= \frac{1}{k}(x_k + (k - 1)\mu_{k-1}) \\= \mu_{k-1} + \frac{1}{k}(x_k - \mu_{k-1}) = \text{Incremental MC Update}$  
> 其中，$\bold{x_k - \mu_{k-1}}\text{為 prediction error}$

- For each $s_t$ visited in an episode with $G_t$
   - $N(s_t) \leftarrow N(s_t) + 1$ ```這個狀態 $s_t$ 又被訪問了一次，累計起來```
   - $V(s_t) \leftarrow V(s_t) + \frac{1}{N(s_t)}(G_t - V(s_t))$ ```就是上面公式換個寫法-moving average```
   - 當面對非平穩問題 ```(Non-stationary problems: 環境規則會隨時間改變時，舊的經驗可能不再可靠)```時，把原本的 $\frac{1}{N}$ 換成一個固定的常數 $\alpha$。
      - *learning rate $\alpha \text{ 取代 } \frac{1}{N(s_t)}$* - 讓 Agent 逐漸「遺忘」很久以前的舊經驗，而更重視最近學到的新東西。

### Temporal Difference (TD) Prediction

- TD 與 MC 的核心差異: **No need to wait for any final outcomes**
   - 可以用在continuous task(沒有明確的回合)
   - Bootstrapping: 從預估中進行預估
  
- 具體方法:
   - TD 在每走一步後，會利用「立即獲得的獎勵加上打折後的下一步預估價值」作為「TD Target（$r_{t+1} + \gamma V(s_{t+1})$）」，以此來取代實際的總回報進行更新
   - 之後再計算TD 目標與當前預估值之間的落差，稱為「TD Error ($r_{t+1} + \gamma V(s_{t+1} )−V(s_t)$）」，並據此進行「逐步備份（step-by-step backup）」的價值更新

> 也就是說: 
> - $V(s_t)\text{不是被完整的 }G_t (\text{回合總回報平均})\text{所推得、更新出來的}$  
> - 在TD，$V(s_t)\text{是被 }R_t + \gamma V(s_{t+1})\bold{的預估}\text{逐漸更新、推得}$  
> 公式:$V(s_t) \leftarrow V(s_t) + \alpha \big[\overbrace{\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t)]}^{\text{TD error}}$

### Algo detail of Temporal Difference

``` c
Algorithm: Tabular TD(0) for Estimating vπ
// 目的： 在已知 policy π 的情況下，透過與環境互動逐步估計每個 state 的 value V(s)

Require: // 輸入: 一個policy π
   π  → 已經存在的 policy（state → action 的機率分布）

Ensure: // 輸出: value finction V(s) 會逐漸逼近在 policy π 下的真實 state value
   V ≈ vπ


Initialize V(s) arbitrarily (e.g., V(s) = 0, ∀ s ∈ S)
   // 為所有 state 初始化 value table, 最常見做法是全部設為 0

repeat  // 開始進行多個 episode，每個 episode 是一次完整遊戲或一次完整環境互動

   Initialize state s // 開始新的回合: 重置環境 + 取得起始 state
   repeat //在此 episode 中持續互動直到 terminal state
      Select action a from π(a|s) // 根據 policy π 選擇 action. π(a|s) 表示在 state s 選 action a 的機率
      Take action a, observe reward r and next state s'' // 與環境互動- 執行 action a 並得到： reward 與 s' → 下一個 state
      V(s) ← V(s) + α [ r + γV(s'' ) − V(s) ] // TD(Temporal Difference) 更新公式
      s ← s'' // 將 state 更新為下一個 state，agent 會繼續從新 state 行動

   until s is terminal //若到達終止 state (terminal state)，則 episode 結束

until value estimates converge //不斷重複多個 episode ，直到 V(s) 不再顯著改變

```

### Comparison between MD & TD
- MC method
   - $G_t$ is obtained based on the actual returns - **unbaised**
   - $G_t$ depends on many $(s,a,r)$ pairs - **higher variance**
- TD method
   - TD target $V(s)$ is based on estimated values - **biased**
   - TD target $V(s)$ depends on one $(s,a,r)$ pair - **lower varianced**

### Multi-Step TD method

- 除了每做一步就估計一次 value function, 也可以 **多做幾步再做估計**
- 用 n-step 的return 作為 TD target
> 考慮 $n$ 步做一次計算:  
> $\text{N-step return: } G_t^n = \sum\limits_{j = 0}^{n}\gamma^nV(S_{t+n})$   
> E.g. *3-step return:* $r_t + \gamma r_{t+1} + \gamma^2r_{t+2} + \gamma^3 r_{t+3}$  
> E.g. *$\infty\text{-step return: }$*$G_t^{\infty} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^T r_{t+T} = \textbf{MC return}$

### $\lambda$-Return
- 有時候，我們可以同時考慮多種不同步數 *multi-step return*
- 利用 *指數加權*，將所有可能的 $n$ 步回報進行加權平均
> 公式: $G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^n$  
> $\lambda\text{(衰減因子): 決定給予}\textbf{遠期回報}\text{多少權重}$  
> 更新機制: $V(s_t) \leftarrow V(s_t) + \alpha(G_t^\lambda - V(s_t))$


## Model-Free Control

- 目標: 在完全未知的環境中，單憑**互動經驗**來找出最佳策略$\pi^*$

> Recap: 在模型可知的環境中，我們利用DP求算*value function* 並且透過*greedy principle*每次選擇最有價值的action做為策略改動。  
> 上述的計算價值(*evaluation*)、選擇動作(*improvement*)、計算價值、更新動作 ...稱為**策略迭代(Policy Iteration)**  
> : $\pi_0 \rightarrow Q_{\pi0} \rightarrow \pi_1 \rightarrow Q_{\pi1} \rightarrow \dots \pi_* \rightarrow Q^*$

### Monte-Carlo Control
- 在一個 *Monte-Carlo Prediction*回合後，我們可以得到策略中採取的一個回合行動軌跡的return，作為其action value 的 **估算**
- 但是，如果使用*greedy principle*會導致agent陷入local optimal而沒有察覺到更佳解  
```也就是說，agent 每次都會選擇已經探勘、知曉action value的action而不選擇其他在policy但尚未知道value的action```

- 原因: *model-free*無法知道所有action/states的 value，需要有**挖掘新action的設計**
- Solution: $\epsilon\text{-Greedy Policy} \rightarrow \text{引入一點隨機性}$

#### $\epsilon$-Greedy Exploration
- 有 $1 - \epsilon$ 的機率選擇當前最好的$\text{action }a^*$
- 有 $\epsilon$ 的機率**隨機**sample action a

> $$\text{policy } \pi = \left\{ \begin{aligned}
1-\lambda & &\text{if } a^* = greedy(Q(s,a)) \\
\lambda & & \text{otherwise}
\end{aligned}\right.$$

### Temporal Difference(TD) Control

- 在 **$\epsilon$-greedy policy**的前提下，TD prediction可以有兩種方式進行*policy improvement* :
   1. On-policy(在線策略): 
      * Agent 遵循目前的策略 $\pi$ 進行探索，並直接根據這些實際經歷來優化該策略。
      * > 邊做邊學，學的就是當前正在做的這套動作。
   2. Off-policy (離線策略)：
      * Agent 擁有一套用於探索環境的「行為策略」*（較具隨機性）*，但它學習的是另一套「目標策略」*（理論上的最優策略）*。
      * > 參考別人的做法或想像最完美的做法來學習，而不一定要親自執行該最優動作。

### On-Policy v.s. Off-Policy
在TD control的情況下，有兩種policy:
1. **Target Policy $\pi$** : 當前agent正在採取的策略 - ```「正在學習、評估，並試圖最佳化」的策略```
2. **Behavior Policy $\mu$**: Agent用來選擇action的策略 - ```用來「產生互動經驗」的策略```
> - On-policy learning : $\pi = \mu$
> - Off-policy learning: $\pi \not ={\mu}$

- Off-policy的優點:
   1. 可以重複利用舊有的、或是純粹用於隨機探索的歷史經驗數據來學習
   2. 可以學習人類的遊玩紀錄（Human replay data）
   3. 能夠利用專家示範（Expert demonstrations），這能大幅加速Agent在複雜環境中的學習速度
- Off-policy需要面對的問題: **Importance Sampling**
   - 因為選擇策略與學習策略不同不能直接拿 μ 的經驗來更新 π，需要考慮**在一個策略之下，另外一個策略的動作有多少價值** (```Estimating the expected values of distribution p by samples from distribution q```)
   - > $\text{令}p(x)\text{為目標學習策略}\pi\text{之下某個action x 的機率 ;} q(x) \text{為選擇策略 之下 action x 的機率}$
   - > $\text{令}f(x)\text{為該action的 return}$
   - > 可推導公式: $\begin{aligned}\mathbb{E}_{x\sim p}[f(x)] &= \sum p(x)f(x) \\ & =\sum q(x)\frac{p(x)}{q(x)}f(x) \\ & =\mathbb{E}_{x\sim q}[\frac{p(x)}{q(x)}f(x)]\end{aligned}$
   - > 其中 $ $\frac{p(x)}{q(x)}$$被稱為*重要性權重* 或 *似然比（Likelihood Ratio*):<li> 權重> 1：代表行為策略很少執行的動作，但在目標策略中非常重要。此時必須放大該樣本的回報貢獻。</li><li>權重 < 1：代表行為策略頻繁執行，但在目標策略中並不常見的動作。此時必須縮小該樣本的影響</li><li>權重 = 0：若行為策略執行了目標策略完全不會採取的動作，則該樣本對評估目標策略沒有貢獻。</li>

#### Importance Sampling for Off-Policy Learning
1. Monte Carlo Target
   - MC 方法是基於走完完整回合的真實總回報（$G_t$）來進行評估。
   - 因此，為了將 μ 產生的回報轉換為 π 的預期回報，必須將從當前時間點 t 一直到回合結束 T 之間，「**每一個步驟的動作機率比值」全部連乘起來**
   - > 修正公式: $G_t^{\pi/\mu} = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\frac{\pi(a_{t+1}|s_{t+1})}{\mu(a_{t+1}| s_{t+1})}\dots \frac{\pi(a_T|s_T)}{\mu(a_T|s_T)}G_t$
   - > 修正公式: $V(s_t)\leftarrow V(s_t) + \alpha(G_t^{\pi/\mu}-V(s_t))$
   - **問題:** 如果行為策略 μ 與目標策略 π 差異很大，這些機率比值的連乘效應會使重要性權重變得極端巨大。 $\Rightarrow$ **少數極端的樣本會獲得極高的權重**，導致學習過程的*變異數（Variance*暴增，造成價值更新劇烈震盪且收斂速度變得非常緩慢
2. TD Target
   - 因為TD 採用了Bootstrapping(走一步算一步)，其原本的更新目標僅依賴 **「當下的獎勵加上下一步的預估價值」** (*i.e. $r_t + \gamma V(s_{t+1})$*)，所以修正也只需要**針對「當下這單一個步驟」的動作機率進行重要性採樣校正**
   - > 修正公式: $\text{TD}_{\text{target}} = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}(r_t + \gamma V(s_{t+1}))$
   - > 修正公式: $V(s_t)\leftarrow V(s_t) + \alpha(\text{TD}_\text{target} - V(s_t))$
   - 變異數顯著小於MC target


### Sarsa

**On-Policy**

#### Mechanism  
1. Agent 在狀態 $s$ 執行動作 $a$。
2. 觀察到獎勵 $r$ 並進入新狀態 $s'$。
3. **Agent 根據「目前的策略」預先決定在 $s'$ 會採取的下一步動作 $a'$。**
4. 使用 $Q(s', a')$ 作為 TD Target 來更新 $Q(s, a)$。

> On-policy: 在更新value時，會考慮到自己接下來真的會去做的動作（包含可能的錯誤或探索）。

#### Formula
**名稱由來: 每次更新考慮 $\bold{(s_t,a_t,r_{t+1}, s_{t+1},a_{t+1})}$**
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$$


### Q-learning

**Off-policy**
#### 特點: 透過$\epsilon$-greedy decision避開了importance sampling的修正問題
#### Mechanism
1. Agent 在狀態 $s$ 執行動作 $a$。
2. 觀察到獎勵 $r$ 並進入新狀態 $s'$。
3. **在更新時，它不關心自己下一步「實際」會做什麼，而是查找在 $s'$ 狀態下，所有可能動作中價值最高的那一個 $\max_{a'} Q(s', a')$。**
4. 即使 Agent 下一步因為隨機探索而走錯路，它在更新 $Q(s, a)$ 時仍然假設自己會走最正確的那條路。
#### Formula

$$
Q(s, a) \leftarrow Q(\underbrace{s, a}_{\text{From }\mu}) + \alpha [\underbrace{r}_{\text{From }\mu} + \gamma \max_{a'} Q(\underbrace{s'}_{\text{From }\mu}, \overbrace{a'}^{\text{from }\pi}) - Q(s, a)]
$$


### Comparison between SARSA & Q-Learning

| 特性 | SARSA (On-policy) | Q-learning (Off-policy) |
| --- | --- | --- |
| **更新依據** | 下一個實際執行的動作 $a'$ | 下一個可能的最佳動作 $\max a'$ |
| **收斂目標** | 當前策略的價值 | 最優策略的價值 |
| **風險偏好** | 較保守，會避開高風險路徑 | 較激進，傾向尋找理論最短路徑 |
| **適用場景** | 在線學習，需考慮執行過程的安全 | 離線學習，或尋找極致最優解 |

