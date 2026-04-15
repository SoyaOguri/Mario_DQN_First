import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from collections import deque
import os

# ==========================================
# 🌟 フレームスキップの仕組み (4フレーム長押し)
# ==========================================
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# --- 1. 脳みその設計図 (CNNモデル) ---
def build_brain(action_size):
    inputs = layers.Input(shape=(84, 84, 4))
    x = layers.Rescaling(1./255)(inputs) # ピクセル値を0〜1に正規化
    x = layers.Conv2D(32, kernel_size=8, strides=4, activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    action_values = layers.Dense(action_size, activation='linear')(x)
    model = models.Model(inputs=inputs, outputs=action_values)
    
    # 安定した学習のためのHuber Lossを使用
    model.compile(loss=losses.Huber(), optimizer=optimizers.Adam(learning_rate=0.00025))
    return model

# ==========================================
# 2. AIプレイヤー (MarioAgent)
# ==========================================
class MarioAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.brain = build_brain(action_size)
        
        # Target Networkの追加
        self.target_brain = build_brain(action_size) 
        self.update_target_network() # 最初は同じ重みにしておく
        
        # ※もしPCのメモリ(RAM)不足で落ちる場合は、ここを50000等に減らしてください
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99999 # 🌟最終調整: ランダム率の減衰を早める(スプリント設定)
        self.batch_size = 32

        self.burnin = 100000
        self.learn_every = 3
        self.sync_every = 10000 # 何歩ごとにTarget Networkを更新するか
        self.curr_step = 0

    # Target Networkを最新の脳みそと同期する関数
    def update_target_network(self):
        self.target_brain.set_weights(self.brain.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = np.squeeze(np.array(state))
        if state_tensor.ndim == 3 and state_tensor.shape[0] == 4:
            state_tensor = np.transpose(state_tensor, (1, 2, 0))
            
        state_tensor = np.expand_dims(state_tensor, axis=0)
        act_values = self.brain.predict(state_tensor, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.curr_step += 1  # 記録するたびにカウンターを進める

    def learn(self):
        if self.curr_step < self.burnin:
            return
        
        # 一定ステップごとにTarget Networkを同期
        if self.curr_step % self.sync_every == 0:
            self.update_target_network()

        if self.curr_step % self.learn_every != 0:
            return
            
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        if states.ndim == 4 and states.shape[1] == 4:
            states = np.transpose(states, (0, 2, 3, 1))
            next_states = np.transpose(next_states, (0, 2, 3, 1))

        # 現在のQ値はbrainから、目標のQ値はtarget_brainから計算する
        targets = self.brain.predict(states, verbose=0)
        next_q_values = self.target_brain.predict(next_states, verbose=0)

        for i, (s, action, reward, ns, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(next_q_values[i])
        
        self.brain.fit(states, targets, epochs=1, verbose=0)

# ==========================================
# 3. メインループ
# ==========================================
def main():
    print(">>> マリオの世界を構築中... (究極完成版)")
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    
    # 🌟 最終調整: 行動を7種類に拡張して、マリオに緩急と自由を与える
    CUSTOM_MOVEMENT = [
        ['NOOP'],             # 行動0: 何もしない（立ち止まる・ブレーキ）
        ['right'],            # 行動1: 右歩き（微調整用）
        ['right', 'A'],       # 行動2: 右ジャンプ
        ['right', 'B'],       # 行動3: 右ダッシュ
        ['right', 'A', 'B'],  # 行動4: 右ダッシュジャンプ
        ['left'],             # 行動5: 左歩き（助走をつけるため）
        ['left', 'A']         # 行動6: 左ジャンプ
    ]
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    agent = MarioAgent(env.action_space.n)
    episodes = 10000 

    for e in range(episodes):
        print(f"\n--- エピソード {e+1} 開始 ---")
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            step_count += 1

            # ログ表示
            if step_count % 500 == 0: 
                status = "準備運動中🏃" if agent.curr_step < agent.burnin else "猛勉強中🧠"
                print(f"  現在 {step_count}歩 (総計 {agent.curr_step}歩) [{status}] ランダム率: {agent.epsilon*100:.1f}%")

            # ランダム率の低下（準備運動が終わってから開始）
            if agent.curr_step > agent.burnin and agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

        print(f"✅ エピソード {e+1}/{episodes} 終了! 合計報酬: {total_reward}")

        # 100エピソードごとにモデルを保存
        if (e + 1) % 100 == 0:
            save_name = f'mario_model_ep{e+1}.h5'
            agent.brain.save(save_name)
            print(f"📂 中間セーブ完了: {save_name} を保存しました。")

    agent.brain.save('mario_model_final.h5')
    print("\n🎉 すべての学習が完了しました.最終モデルを 'mario_model_final.h5' に保存しました。")

    env.close()

if __name__ == '__main__':
    main()