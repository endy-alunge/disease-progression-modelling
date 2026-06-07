import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_transition_matrix(transition_matrix, state_names, title="Transition Matrix"):
    n_states = len(transition_matrix)
    
    # Diagonal dominance (state persistence)
    diag_dominance = np.mean([transition_matrix[i, i] for i in range(n_states)])
    
    # Determinism - how predictable transitions are
    determinism = np.mean([np.max(transition_matrix[i]) for i in range(n_states)])
    
    # Entropy analysis
    entropies = []
    for i in range(n_states):
        p = transition_matrix[i]
        entropy = -np.sum(p * np.log(p + 1e-10))
        entropies.append(entropy)
    
    avg_entropy = np.mean(entropies)
    max_entropy = np.log(n_states)
    normalized_entropy = avg_entropy / max_entropy
    
    # Print results
    print(f"\n  {title} Analysis:")
    print(f"    Diagonal dominance (self-loop probability): {diag_dominance:.3f}")
    print(f"    Transition determinism: {determinism:.3f}")
    print(f"    Average entropy: {avg_entropy:.3f} / {max_entropy:.3f}")
    print(f"    Normalized entropy: {normalized_entropy:.3f}")
    
    if normalized_entropy < 0.3:
        print(f"    → LOW ENTROPY: Transitions are highly predictable")
    elif normalized_entropy < 0.6:
        print(f"    → MODERATE ENTROPY: Some uncertainty in transitions")
    else:
        print(f"    → HIGH ENTROPY: Transitions are near-random")
    
    return {
        'diagonal_dominance': diag_dominance,
        'determinism': determinism,
        'avg_entropy': avg_entropy,
        'normalized_entropy': normalized_entropy,
        'entropies': entropies
    }


def analyze_sequence_determinism(sequences, max_history=5):
    history_next = defaultdict(set)
    
    for seq in sequences:
        for t in range(len(seq) - 1):
            # Use limited history length
            start = max(0, t - max_history + 1)
            history = tuple(seq[start:t+1])
            history_next[history].add(seq[t+1])
    
    # Count deterministic vs stochastic
    deterministic = sum(1 for next_set in history_next.values() if len(next_set) == 1)
    stochastic = sum(1 for next_set in history_next.values() if len(next_set) > 1)
    total = deterministic + stochastic
    
    determinism_ratio = deterministic / total if total > 0 else 0
    
    print(f"\n  Sequence Determinism Analysis:")
    print(f"    Unique histories: {len(history_next)}")
    print(f"    Deterministic histories: {deterministic} ({determinism_ratio*100:.1f}%)")
    print(f"    Stochastic histories: {stochastic} ({100-determinism_ratio*100:.1f}%)")
    
    if determinism_ratio > 0.95:
        print(f"    → HIGHLY DETERMINISTIC: Same history → same next state")
        print(f"    → This explains perfect model performance")
    elif determinism_ratio > 0.7:
        print(f"    → MODERATELY DETERMINISTIC: Some stochasticity present")
    else:
        print(f"    → STOCHASTIC: Significant uncertainty in transitions")
    
    return {
        'determinism_ratio': determinism_ratio,
        'total_histories': len(history_next),
        'deterministic_count': deterministic,
        'stochastic_count': stochastic
    }


def analyze_hmm_structure(hmm_model, sequences, state_names):
    if hmm_model.model is None:
        print("  HMM not trained yet")
        return None
    
    # Decode hidden states for all sequences
    all_hidden = []
    for seq in sequences:
        obs = np.array(seq).reshape(-1, 1)
        _, hidden = hmm_model.model.decode(obs, algorithm="viterbi")
        all_hidden.extend(hidden)
    
    hidden_counts = np.bincount(all_hidden, minlength=hmm_model.n_hidden)
    total = len(all_hidden)
    
    print(f"\n  HMM Hidden State Analysis:")
    print(f"    Hidden state usage distribution:")
    for i in range(min(8, hmm_model.n_hidden)):
        pct = 100 * hidden_counts[i] / total
        bar = "█" * int(pct / 5)
        print(f"      State {i}: {pct:5.1f}% {bar}")
    
    # Emission analysis
    print(f"\n    Emission patterns (hidden → observation):")
    for i in range(min(5, hmm_model.n_hidden)):
        emissions = hmm_model.model.emissionprob_[i]
        most_likely = np.argmax(emissions)
        confidence = emissions[most_likely]
        obs_name = state_names[most_likely] if most_likely < len(state_names) else str(most_likely)
        print(f"      State {i} → {obs_name} (confidence: {confidence:.3f})")
    
    # Transition analysis
    print(f"\n    Hidden state transitions (top 5):")
    transmat = hmm_model.model.transmat_
    transitions = []
    for i in range(hmm_model.n_hidden):
        for j in range(hmm_model.n_hidden):
            if i != j and transmat[i, j] > 0.05:
                transitions.append((i, j, transmat[i, j]))
    transitions.sort(key=lambda x: x[2], reverse=True)
    for i, j, prob in transitions[:5]:
        print(f"      State {i} → State {j}: {prob:.3f}")
    
    # Calculate hidden state entropy
    hidden_probs = hidden_counts / total
    hidden_entropy = -np.sum(hidden_probs * np.log(hidden_probs + 1e-10))
    max_hidden_entropy = np.log(hmm_model.n_hidden)
    
    print(f"\n    Hidden state entropy: {hidden_entropy:.3f} / {max_hidden_entropy:.3f}")
    
    return {
        'hidden_counts': hidden_counts,
        'hidden_entropy': hidden_entropy,
        'normalized_hidden_entropy': hidden_entropy / max_hidden_entropy
    }


def plot_transition_heatmap(transition_matrix, state_names, title, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=state_names, yticklabels=state_names)
    plt.title(title, fontsize=14)
    plt.xlabel('Next State', fontsize=12)
    plt.ylabel('Current State', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_entropy_bars(entropies, state_names, title, save_path=None):
    plt.figure(figsize=(10, 6))
    colors = ['green' if e < 0.5 else 'orange' if e < 1.0 else 'red' for e in entropies]
    bars = plt.bar(range(len(entropies)), entropies, color=colors, edgecolor='black')
    plt.xticks(range(len(entropies)), state_names[:len(entropies)], rotation=45)
    plt.ylabel('Entropy (nats)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axhline(y=np.log(len(state_names)), color='red', linestyle='--', 
                label=f'Maximum entropy ({np.log(len(state_names)):.3f})')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def run_full_structure_analysis(data, mc_diag, mc_prog, hmm_diag, hmm_prog, save_dir=None):
    print("\n" + "="*70)
    print("STRUCTURAL ANALYSIS: Why Models Achieve High Accuracy")
    print("="*70)
    
    results = {}
    
    # 1. Markov Chain - Diagnosis
    print("\n" + "─"*50)
    print("MARKOV CHAIN - DIAGNOSIS (B/M)")
    print("─"*50)
    mc_diag_results = analyze_transition_matrix(mc_diag.P, mc_diag.class_names, "Diagnosis")
    results['mc_diag'] = mc_diag_results
    
    if save_dir:
        plot_transition_heatmap(mc_diag.P, mc_diag.class_names, 
                                "Markov Chain - Diagnosis Transitions",
                                f"{save_dir}/mc_diag_transition.png")
        plot_entropy_bars(mc_diag_results['entropies'], mc_diag.class_names,
                         "Diagnosis Transition Entropy per State",
                         f"{save_dir}/mc_diag_entropy.png")
    
    # 2. Markov Chain - Progression
    print("\n" + "─"*50)
    print("MARKOV CHAIN - PROGRESSION")
    print("─"*50)
    mc_prog_results = analyze_transition_matrix(mc_prog.P, mc_prog.class_names, "Progression")
    results['mc_prog'] = mc_prog_results
    
    if save_dir:
        plot_transition_heatmap(mc_prog.P, mc_prog.class_names,
                                "Markov Chain - Progression Transitions",
                                f"{save_dir}/mc_prog_transition.png")
    
    # 3. Sequence Determinism
    print("\n" + "─"*50)
    print("SEQUENCE DETERMINISM ANALYSIS")
    print("─"*50)
    diag_determinism = analyze_sequence_determinism(data["diag_seqs"]["train"])
    prog_determinism = analyze_sequence_determinism([p["prog_seq"] for p in data["patients"]["train"]])
    results['diag_determinism'] = diag_determinism
    results['prog_determinism'] = prog_determinism
    
    # 4. HMM Analysis
    print("\n" + "─"*50)
    print("HIDDEN MARKOV MODEL ANALYSIS")
    print("─"*50)
    print("\n  Diagnosis HMM:")
    hmm_diag_results = analyze_hmm_structure(hmm_diag, data["diag_seqs"]["train"], hmm_diag.class_names)
    results['hmm_diag'] = hmm_diag_results
    
    print("\n  Progression HMM:")
    prog_seqs = [p["prog_seq"] for p in data["patients"]["train"]]
    hmm_prog_results = analyze_hmm_structure(hmm_prog, prog_seqs, hmm_prog.class_names)
    results['hmm_prog'] = hmm_prog_results
    
    # 5. Stage-based analysis (if available)
    if "stage_seqs" in data and len(data["stage_seqs"]["train"]) > 0:
        print("\n" + "─"*50)
        print("STAGE TRANSITION ANALYSIS (5 classes)")
        print("─"*50)
        stage_train = data["stage_seqs"]["train"]
        # Create temporary Markov for stages
        from models.markov_chain import MarkovChainModel
        mc_stage = MarkovChainModel(task="stage")
        mc_stage.fit(stage_train)
        stage_results = analyze_transition_matrix(mc_stage.P, mc_stage.class_names, "Stages")
        results['stage'] = stage_results
        results['stage_determinism'] = analyze_sequence_determinism(stage_train)
        
        if save_dir:
            plot_transition_heatmap(mc_stage.P, mc_stage.class_names,
                                   "Markov Chain - Stage Transitions (5 classes)",
                                   f"{save_dir}/stage_transition.png")
    
    return results