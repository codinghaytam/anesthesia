import numpy as np


def _prepare_population(load_data_fn, preprocess_data_fn, generate_dataset_fn, sample_size=100):
    raw_data = load_data_fn("./data/Patients Data.csv")
    cols = ["PatientID", "Sex", "WeightInKilograms", "HeightInMeters", "AgeCategory"]
    df_clean = preprocess_data_fn(raw_data[cols])

    if len(df_clean) > sample_size:
        print(f"\nSampling {sample_size} patients from {len(df_clean)} total...")
        df_clean = df_clean.sample(n=sample_size, random_state=42)

    print(f"Generating Schnider parameters for {len(df_clean)} patients...")
    return generate_dataset_fn(df_clean)


def _print_results(results_df):
    if len(results_df) == 0:
        print("No patients were successfully evaluated.")
        return

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nEvaluated {len(results_df)} patients successfully\n")
    print(results_df[["MDPE", "MDAPE", "Wobble", "Controlled (%)"]].describe())

    best_idx = results_df["MDAPE"].idxmin()
    worst_idx = results_df["MDAPE"].idxmax()

    print(f"\n--- Best Controlled Patient (ID: {results_df.loc[best_idx, 'PatientID']}) ---")
    print(f"  MDPE: {results_df.loc[best_idx, 'MDPE']:.2f}%")
    print(f"  MDAPE: {results_df.loc[best_idx, 'MDAPE']:.2f}%")
    print(f"  Wobble: {results_df.loc[best_idx, 'Wobble']:.2f}%")
    print(f"  Controlled (%): {results_df.loc[best_idx, 'Controlled (%)']:.2f}%")

    print(f"\n--- Worst Controlled Patient (ID: {results_df.loc[worst_idx, 'PatientID']}) ---")
    print(f"  MDPE: {results_df.loc[worst_idx, 'MDPE']:.2f}%")
    print(f"  MDAPE: {results_df.loc[worst_idx, 'MDAPE']:.2f}%")
    print(f"  Wobble: {results_df.loc[worst_idx, 'Wobble']:.2f}%")
    print(f"  Controlled (%): {results_df.loc[worst_idx, 'Controlled (%)']:.2f}%")


def run_saved_dp_evaluation(dp_path, evaluator_cls, load_data_fn, preprocess_data_fn, generate_dataset_fn, sample_size=100):
    print("=" * 60)
    print("LOADING SAVED DP AGENT")
    print("=" * 60)

    if not dp_path.exists():
        print(f"ERROR: Agent file not found at {dp_path}")
        print("Please run the DP training cell first.")
        return None

    try:
        print(f"\nLoading agent from: {dp_path}")
        agent_data = np.load(dp_path)

        print(f"  - Loaded V (value function): shape {agent_data['V'].shape}")
        print(f"  - Loaded policy: shape {agent_data['policy'].shape}")
        print(f"  - Loaded P (transitions): shape {agent_data['P'].shape}")
        print(f"  - Loaded R (rewards): shape {agent_data['R'].shape}")
        print(f"  - Actions available: {agent_data['actions']}")
        print(f"  - Gamma (discount): {agent_data['gamma']}")

        evaluator = evaluator_cls(agent_data["policy"], agent_data["actions"])

        print("\n" + "=" * 60)
        print("EVALUATING ON POPULATION SAMPLE")
        print("=" * 60)

        df_sim = _prepare_population(
            load_data_fn=load_data_fn,
            preprocess_data_fn=preprocess_data_fn,
            generate_dataset_fn=generate_dataset_fn,
            sample_size=sample_size,
        )

        print("\nRunning evaluation simulation (120 min per patient)...")
        results_df = evaluator.evaluate(df_sim)
        _print_results(results_df)
        return results_df

    except Exception as e:
        import traceback

        print(f"ERROR: {e}")
        traceback.print_exc()
        return None


def run_saved_q_evaluation(q_path, evaluator_cls, load_data_fn, preprocess_data_fn, generate_dataset_fn, sample_size=100):
    print("=" * 60)
    print("LOADING SAVED Q-LEARNING AGENT")
    print("=" * 60)

    if not q_path.exists():
        print(f"ERROR: Agent file not found at {q_path}")
        print("Please run the Q-Learning training cell first.")
        return None

    try:
        print(f"\nLoading agent from: {q_path}")
        agent_data = np.load(q_path)

        print(f"  - Loaded Q table: shape {agent_data['Q'].shape}")
        print(f"  - Actions available: {agent_data['actions']}")
        print(f"  - Target BIS: {agent_data['target_bis']}")
        print(f"  - Learning rate (alpha): {agent_data['alpha']}")
        print(f"  - Discount factor (gamma): {agent_data['gamma']}")
        print(f"  - Exploration rate (epsilon): {agent_data['epsilon']}")

        evaluator = evaluator_cls(agent_data["Q"], agent_data["actions"])

        print("\n" + "=" * 60)
        print("EVALUATING ON POPULATION SAMPLE")
        print("=" * 60)

        df_sim = _prepare_population(
            load_data_fn=load_data_fn,
            preprocess_data_fn=preprocess_data_fn,
            generate_dataset_fn=generate_dataset_fn,
            sample_size=sample_size,
        )

        print("\nRunning evaluation simulation (120 min per patient)...")
        results_df = evaluator.evaluate(df_sim)
        _print_results(results_df)
        return results_df

    except Exception as e:
        import traceback

        print(f"ERROR: {e}")
        traceback.print_exc()
        return None


def run_quick_dp_evaluation(dp_path, evaluator_cls, load_data_fn, preprocess_data_fn, generate_dataset_fn, sample_size=50):
    try:
        print("Loading Population Data...")
        df_sim = _prepare_population(
            load_data_fn=load_data_fn,
            preprocess_data_fn=preprocess_data_fn,
            generate_dataset_fn=generate_dataset_fn,
            sample_size=sample_size,
        )

        print("Loading DP Agent...")
        dp_data = np.load(dp_path)
        evaluator = evaluator_cls(dp_data["policy"], dp_data["actions"])

        print("Evaluating on Population...")
        results_df = evaluator.evaluate(df_sim)
        print("\n--- Evaluation Results Summary ---")
        print(results_df[["MDPE", "MDAPE", "Wobble", "Controlled (%)"]].describe())

        best_pt = results_df.loc[results_df["MDAPE"].idxmin()]
        print(f"\nBest Patient: {best_pt.to_dict()}")
        return results_df

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


def run_quick_q_evaluation(q_path, evaluator_cls, load_data_fn, preprocess_data_fn, generate_dataset_fn, sample_size=50):
    try:
        print("Loading Population Data...")
        df_sim = _prepare_population(
            load_data_fn=load_data_fn,
            preprocess_data_fn=preprocess_data_fn,
            generate_dataset_fn=generate_dataset_fn,
            sample_size=sample_size,
        )

        print("Loading Q-Learning Agent...")
        q_data = np.load(q_path)
        evaluator = evaluator_cls(q_data["Q"], q_data["actions"])

        print("Evaluating on Population...")
        results_df = evaluator.evaluate(df_sim)
        print("\n--- Evaluation Results Summary ---")
        print(results_df[["MDPE", "MDAPE", "Wobble", "Controlled (%)"]].describe())

        best_pt = results_df.loc[results_df["MDAPE"].idxmin()]
        print(f"\nBest Patient: {best_pt.to_dict()}")
        return results_df

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None
