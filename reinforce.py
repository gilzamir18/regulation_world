import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from env import HomeostaticEnv

# Configurações de estilo para os gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

class PolicyNetwork(nn.Module):
    """
    Política baseada em distribuição Normal (Gaussian) para ações contínuas.
    A rede neural prediz a média (mu), e o desvio padrão (std) é um parâmetro aprendível independente do estado.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # O logaritmo do std é aprendível, inicializado em 0 (std = 1)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def get_action_and_log_prob(self, obs):
        mu, std = self(obs)
        dist = Normal(mu, std)
        action = dist.sample()
        # Soma os log probs ao longo das dimensões de ação 
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

def compute_returns(rewards, gamma):
    """
    Calcula os retornos descontados G_t de trás para frente.
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def train_reinforce(env, policy, optimizer, num_episodes, gamma=0.99):
    """
    Treina a política usando o algoritmo REINFORCE episódico padrão.
    """
    stats = {
        "episode_length": [],
        "policy_loss_std": [],
        "avg_reward": []
    }

    iterator = range(num_episodes)
    if num_episodes > 50:
        iterator = tqdm(iterator, desc="Training", leave=False)

    for episode in iterator:
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        
        done = False
        truncated = False
        
        # 1. Coletar trajetória
        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob = policy.get_action_and_log_prob(obs_tensor)
            
            action_np = action.squeeze(0).detach().numpy()
            
            next_obs, reward, done, truncated, _ = env.step(action_np)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs
            
        # 2. Calcular retornos e atualizar
        returns = compute_returns(rewards, gamma)
        returns_tensor = torch.FloatTensor(returns)
        if returns_tensor.shape[0] > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            
        policy_loss = []
        for log_prob, G in zip(log_probs, returns_tensor):
            policy_loss.append(-log_prob * G)
            
        policy_loss = torch.cat(policy_loss).sum()
        
        loss_components = torch.cat([-lp * g for lp, g in zip(log_probs, returns_tensor)])
        loss_std = loss_components.std().item() if loss_components.shape[0] > 1 else 0.0

        optimizer.zero_grad()
        if returns_tensor.shape[0] > 1:
            policy_loss.backward()
            optimizer.step()
        
        # 3. Salvar métricas do episódio
        stats["episode_length"].append(env.step_count)
        stats["policy_loss_std"].append(loss_std)
        stats["avg_reward"].append(np.mean(rewards))

    return stats

def run_experiment(rewarding_variant, seed, num_vars=100, num_episodes=500, max_steps=1000):
    print(f"Starting {rewarding_variant} - Seed {seed}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    env = HomeostaticEnv(
        num_vars=num_vars, 
        rewarding=rewarding_variant, 
        reward_scale=0.1, 
        max_steps=max_steps, 
        random_safety_zone=True
    )
    env.reset(seed=seed)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=64)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3) 
    
    stats = train_reinforce(env, policy, optimizer, num_episodes=num_episodes)
    
    df = pd.DataFrame(stats)
    df["episode"] = df.index + 1
    df["variant"] = rewarding_variant
    df["seed"] = seed
    
    eval_window = min(50, num_episodes)
    final_ep_len = df["episode_length"].tail(eval_window).mean()
    final_loss_std = df["policy_loss_std"].tail(eval_window).mean()
    
    resultado_final = {
        "variant": rewarding_variant,
        "seed": seed,
        "final_mean_episode_length": final_ep_len,
        "final_mean_policy_loss_std": final_loss_std
    }
    
    return df, resultado_final

def plot_learning_curves(df_all, base_dir="figures"):
    plt.figure(figsize=(10, 6))
    
    df_all['smoothed_ep_length'] = df_all.groupby(['variant', 'seed'])['episode_length'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

    sns.lineplot(
        data=df_all, 
        x='episode', 
        y='smoothed_ep_length', 
        hue='variant', 
        errorbar='sd'
    )
    plt.title("Episode Length during Training")
    plt.xlabel("Episodes")
    plt.ylabel("Average Episode Length")
    plt.legend(title="Variant (Reward)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "learning_curve_episode_length.png"))
    plt.savefig(os.path.join(base_dir, "learning_curve_episode_length.pdf"))
    plt.close()

def plot_final_metrics(df_results, base_dir="figures"):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_results, x='variant', y='final_mean_episode_length')
    sns.stripplot(data=df_results, x='variant', y='final_mean_episode_length', color='black', alpha=0.5)
    plt.title("Structural Ablation: Final Episode Length")
    plt.xlabel("Variant")
    plt.ylabel("Episode Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "boxplot_final_episode_length.png"))
    plt.savefig(os.path.join(base_dir, "boxplot_final_episode_length.pdf"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_results, x='variant', y='final_mean_policy_loss_std')
    sns.stripplot(data=df_results, x='variant', y='final_mean_policy_loss_std', color='black', alpha=0.5)
    plt.title("Structural Ablation: Final Policy Loss Std")
    plt.xlabel("Variant")
    plt.ylabel("Policy Loss Std")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "boxplot_final_policy_loss_std.png"))
    plt.savefig(os.path.join(base_dir, "boxplot_final_policy_loss_std.pdf"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="REINFORCE Ablation Study on HomeostaticEnv")
    parser.add_argument("--num-vars", type=int, default=100, help="Dimensionalidad (default: 100).")
    parser.add_argument("--num-episodes", type=int, default=150, help="Número de episódios de treino por seed.")
    parser.add_argument("--seeds", type=int, default=5, help="Quantidade de seeds por condição.")
    parser.add_argument("--test-mode", action="store_true", help="Modo teste.")
    args = parser.parse_args()

    variants = [
        "OR_full",
        "OR_no_safety_zone",
        "OR_unbounded",
        "OR_no_maintenance_reward"
    ]

    seeds_list = list(range(42, 42 + args.seeds))
    num_episodes = args.num_episodes
    if args.test_mode:
        print("TEST MODE ACTIVATED")
        variants = ["OR_full", "OR_no_safety_zone"]
        seeds_list = [42, 43]
        num_episodes = 5

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    all_progress_logs = []
    all_final_results = []

    for variant in variants:
        for seed in seeds_list:
            df_prog, res_final = run_experiment(
                rewarding_variant=variant, 
                seed=seed, 
                num_vars=args.num_vars, 
                num_episodes=num_episodes
            )
            all_progress_logs.append(df_prog)
            all_final_results.append(res_final)

    # Consolidar os dados
    df_all_progress = pd.concat(all_progress_logs, ignore_index=True)
    df_all_results = pd.DataFrame(all_final_results)

    # Salvar em arquivos
    print("\nSaving metrics and summary tables...")
    df_all_progress.to_csv("results/training_metrics.csv", index=False)
    df_all_results.to_csv("results/final_results_summary.csv", index=False)

    # Gerar os gráficos
    print("Generating plots...")
    plot_learning_curves(df_all_progress, base_dir="figures")
    plot_final_metrics(df_all_results, base_dir="figures")

    # Mostrar Tabela-Visão Geral no Console e num relatório TXT
    summary_table = df_all_results.groupby('variant').agg(
        Mean_Episode_Length=('final_mean_episode_length', 'mean'),
        Std_Episode_Length=('final_mean_episode_length', 'std'),
        Mean_Loss_Std=('final_mean_policy_loss_std', 'mean')
    ).reset_index()

    report = "### Ablation Study Summary ###\n\n"
    report += summary_table.to_string(index=False)
    report += "\n\nFiles saved in ./results/ and ./figures/"
    print("\n" + report)
    
    with open("results/ablation_report.txt", "w") as f:
        f.write(report)
        f.write("\n\nExecution Conditions:\n")
        f.write(f"Variables: {args.num_vars}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Seeds: {len(seeds_list)}\n")

if __name__ == "__main__":
    main()
