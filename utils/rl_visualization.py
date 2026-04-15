import numpy as np
import matplotlib.pyplot as plt


def evaluate_policy(
    agent,
    n_episodes,
    ep_len,
    bis_target_range,
    bis_fn,
    features_fn,
    pk_step_fn,
    actions,
    action_interval=1,
):
    rewards = []
    for _ in range(n_episodes):
        target = np.random.uniform(*bis_target_range)
        c0 = np.random.uniform(0.5, 4.0)
        state = np.array([c0, c0 * 0.3, c0 * 0.1, c0 * 0.5], dtype=float)

        err_prev = bis_fn(state[3]) - target
        ep_reward = 0.0
        action_idx = agent.select_action(features_fn(err_prev, 0.0), training=False)
        for step in range(ep_len):
            state = np.maximum(pk_step_fn(state, actions[action_idx]), 0.0)
            err = bis_fn(state[3]) - target
            ep_reward += -abs(err)

            if (step + 1) % action_interval == 0:
                action_idx = agent.select_action(
                    features_fn(err, err - err_prev), training=False
                )
            err_prev = err

        rewards.append(ep_reward)

    print(
        f"\nEvaluation: Mean Reward = {np.mean(rewards):.2f} +- {np.std(rewards):.2f}"
    )
    return rewards


def plot_training_curves(
    rewards, losses, save_path="images/training_curves.png", window=50
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rewards, alpha=0.6, linewidth=1, label="Episode Reward")
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax1.plot(
        range(window - 1, len(rewards)),
        smoothed,
        "r-",
        linewidth=2,
        label=f"Moving Avg ({window})",
    )
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Total Reward", fontsize=12)
    ax1.set_title("Training Rewards", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(losses, alpha=0.6, linewidth=1, label="Bellman Loss")
    smoothed_loss = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax2.plot(
        range(window - 1, len(losses)),
        smoothed_loss,
        "g-",
        linewidth=2,
        label=f"Moving Avg ({window})",
    )
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Training Loss (MSE)", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_bis_trajectory(
    agent,
    ep_len,
    target,
    bis_target_range,
    bis_fn,
    features_fn,
    pk_step_fn,
    actions,
    save_path="images/bis_trajectory.png",
):
    c0 = 1.0
    state = np.array([c0, c0 * 0.3, c0 * 0.1, c0 * 0.5], dtype=float)

    bis_vals = []
    actions_taken = []
    times = []

    err_prev = bis_fn(state[3]) - target

    for t in range(ep_len):
        feat = features_fn(err_prev, 0.0)
        action_idx = agent.select_action(feat, training=False)

        bis_vals.append(bis_fn(state[3]))
        actions_taken.append(actions[action_idx])
        times.append(t)

        state = np.maximum(pk_step_fn(state, actions[action_idx]), 0.0)
        err = bis_fn(state[3]) - target
        err_prev = err

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(times, bis_vals, "b-", linewidth=2, label="BIS (Patient)")
    ax1.axhline(
        y=target,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Target BIS ({target:.1f})",
    )
    ax1.fill_between(
        times,
        bis_target_range[0],
        bis_target_range[1],
        alpha=0.2,
        color="green",
        label="Target Range",
    )
    ax1.set_ylabel("BIS Index", fontsize=12)
    ax1.set_title(
        "Patient BIS Evolution During Controlled Infusion",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    ax2.plot(times, actions_taken, "g-", linewidth=2, label="Infusion Rate")
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Infusion Rate (ml/min)", fontsize=12)
    ax2.set_title("Agent Control Action", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 6])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_q_values_heatmap(n_samples, q_max_fn, save_path="images/q_values_heatmap.png"):
    err_range = np.linspace(-50, 50, n_samples)
    derr_range = np.linspace(-10, 10, n_samples)

    q_max = np.zeros((len(derr_range), len(err_range)))
    for i, derr in enumerate(derr_range):
        for j, err in enumerate(err_range):
            q_max[i, j] = q_max_fn(err, derr)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        q_max,
        extent=[-50, 50, -10, 10],
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        interpolation="bilinear",
    )
    ax.set_xlabel("BIS Error", fontsize=12)
    ax.set_ylabel("BIS Error Rate (15s change)", fontsize=12)
    ax.set_title("Learned Value Function: max_a Q(s,a)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Max Q-value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_policy_heatmap(
    agent, n_samples, features_fn, actions, save_path="images/policy_heatmap.png"
):
    err_range = np.linspace(-50, 50, n_samples)
    derr_range = np.linspace(-10, 10, n_samples)

    policy = np.zeros((len(derr_range), len(err_range)))
    for i, derr in enumerate(derr_range):
        for j, err in enumerate(err_range):
            feat = features_fn(err, derr)
            a = agent.select_action(feat, training=False)
            policy[i, j] = actions[a] * 60

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        policy,
        extent=[-50, 50, -10, 10],
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="bilinear",
    )
    ax.set_xlabel("BIS Error", fontsize=12)
    ax.set_ylabel("BIS Error Rate (15s change)", fontsize=12)
    ax.set_title(
        "Learned Policy: Infusion Rate (ml/min)", fontsize=14, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Action (ml/min)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_action_distribution(
    agent,
    n_episodes,
    ep_len,
    bis_target_range,
    bis_fn,
    features_fn,
    pk_step_fn,
    actions,
    save_path="images/action_distribution.png",
):
    action_counts = np.zeros(len(actions))

    for _ in range(n_episodes):
        target = np.mean(bis_target_range)
        c0 = np.random.uniform(0.5, 2.0)
        state = np.array([c0, c0 * 0.3, c0 * 0.1, c0 * 0.5], dtype=float)
        err_prev = bis_fn(state[3]) - target

        for _ in range(ep_len):
            feat = features_fn(err_prev, 0.0)
            action_idx = agent.select_action(feat, training=False)
            action_counts[action_idx] += 1
            state = np.maximum(pk_step_fn(state, actions[action_idx]), 0.0)
            err_prev = bis_fn(state[3]) - target

    fig, ax = plt.subplots(figsize=(10, 6))
    action_labels = [f"{actions[i] * 60:.2f}" for i in range(len(actions))]
    bars = ax.bar(
        action_labels, action_counts, color="steelblue", alpha=0.8, edgecolor="black"
    )
    ax.set_xlabel("Infusion Rate (ml/min)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Action Distribution in Learned Policy", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_simple_episode(history_bis, history_actions, target=50.0, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(history_bis, color="blue", label="BIS Mesure")
    ax1.axhline(target, color="red", linestyle="--", label=f"Cible (BIS={target:.0f})")
    ax1.set_ylabel("Indice Bispectral (BIS)")
    ax1.set_title("Performance de l'agent RL (Dernier Episode)")
    ax1.legend()

    ax2.step(
        range(len(history_actions)),
        history_actions,
        color="green",
        label="Infusion (ml/min)",
    )
    ax2.set_ylabel("Taux d'infusion")
    ax2.set_xlabel("Temps (secondes)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
