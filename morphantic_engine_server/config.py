class Config:

    def __init__(self):
        self.aea_core = {}
        self.benchmarking = {}
        self.system = {}
        self.output = {}  

    abs_tol = 1e-8
    auto_low_budget_tuning = True
    base_transpose_prob = 1.0
    breeding_summit_interval = 15
    causal_direction_min_samples = 2
    causal_mutagen_influence = 0.3
    causal_tapestry_decay_rate = 0.1  
    causal_tapestry_generation_window = 2
    champion_linesearch_fitness_threshold = 0.05
    champion_linesearch_regular_interval = 4
    champion_linesearch_stagnation_threshold = 5
    champion_linesearch_step_scale = 0.1
    circadian_day_fraction_highd = 0.45
    circadian_day_fraction_static = 0.25
    circadian_period = 100
    circadian_phase_jitter = 0.1
    cmaes_diag_cov = True
    cmaes_enabled = True
    cmaes_gens_cap = 2
    cmaes_lambda_cap = 16
    cmaes_min_budget = 6
    cmaes_trust_radius_fraction = 0.004
    cmaes_use_surrogate = True
    coarse_step_fraction_of_range = 0.005
    conservative_factor_decrease = 0.98
    conservative_factor_increase = 1.02
    default_percentile_weight = 0.0
    default_stress = 0.3
    disable_causal_queries = False
    dream_fitness_threshold = 0.1
    dream_interval = 15
    dream_max_fraction = 0.25
    dream_sigma_scale = 0.2
    dream_stagnation_multiplier = 10
    dream_stagnation_threshold = 5
    early_stop_patience = 1000000000
    enable_aggressive_dreams = True
    enable_batched_eval = False   
    enable_champion_linesearch = True
    enable_coarse_first = False
    enable_dream = True
    enable_epigenetic = True
    enable_epsilon_memoization = True
    enable_exploit_queue = True
    enable_immune = False
    enable_ode_skip = True
    enable_one_fifth_rule = False
    enable_parallel_islands = True
    enable_plateau_unlock = True
    epsilon_step_fraction_of_range = 0.001
    exploit_budget_initial = 16
    exploit_budget_max = 64
    exploit_budget_min = 8
    exploit_budget_penalty_factor = 0.5
    exploit_budget_reward_factor = 1.2
    exploit_queue_allow_mid_phase = True
    exploit_queue_circadian_day_only = True
    exploit_queue_steps = 8
    exploit_queue_top_k = 6
    fast_disable_causal_queries = True
    fast_disable_dream = True
    fast_mode = False
    fast_reduce_detailed_logging = True
    feature_dim = 30
    force_exploit_strategy = None
    gradient_checkpointing = True
    heavy_tail_mutation_prob = 0.0
    heavy_tail_scale = 0.0
    hidden_dim = 128
    high_dim_threshold = 30
    high_fitness_threshold_norm = 0.8
    high_percentile = 90.0
    horizontal_transfer_prob = 0.2
    immune_eviction_generations = 10000
    immune_signature_bits = 256
    intervention_multiplier = 0.05
    island_roles = ["raw", "trust_region", "rescale", "explore", "decomp_or_indicator"]
    island_workers = 0
    late_phase_min_weight = 0.0
    late_phase_sigma_factor = 0.02
    late_phase_start = 0.85 
    late_phase_stress_threshold = 0.85
    low_fitness_threshold_norm = 0.2
    low_percentile = 20.0
    max_depth = 1.5
    max_generations = 200
    max_genes_per_clone = 30
    microchimerism_interval = 12
    
    migration_interval = 15
    min_depth = 0.1
    min_weight_when_history_bad = 0.15
    n_islands = 5
    neural_mutation_probability = 0.05
    neural_mutation_strength = 0.05
    ode_atol = 0.0001
    ode_rtol = 0.01
    ode_solver = "euler"
    ode_time_points = 2
    percentile_window = 50
    phase_crisis_consecutive_generations = 5
    phase_crisis_locked_threshold = 0.65
    phase_crisis_unlock_fraction = 0.2
    phase_transition_detector_window_size = 10
    phenotype_min_mutation = 0.001
    phenotype_scale_factor = 0.25
    pnn_exploit_budget = 256.0
    pnn_hard_unlock_generations = 12
    pnn_hard_unlock_threshold = 0.001
    pnn_improvement_threshold = 0.01
    pnn_stagnation_generations = 8
    pnn_stagnation_penalty_multiplier = 3.0
    pnn_stagnation_penalty_threshold = 5
    pnn_stress_unlock_threshold = 0.8
    pnn_unlock_limit_per_gen = 6
    pop_size = 60
    pure_random_parents = False
    rel_tol = 0.001
    sigma_init_scale = 0.2
    solution_mutation_strength = 0.05
    stagnation_restart_fraction = 0.2
    stagnation_restart_generations = 1000000000
    stress_enable_fallback = True
    stress_immune_explore_trials = 4
    stress_immune_max_cache = 128   
    stress_immune_sigma_threshold = 1.0
    stress_multiplier = 1.0
    stress_relax = 0.45
    stress_threshold = 0.83
    stress_v2_relative_stagnation_weight = 0.7
    stress_v2_time_stagnation_period = 20.0
    stress_v2_time_stagnation_weight = 0.3
    
    surrogate_sampling_nfe = 20
    tapestry_bootstrap_generations = 5
    tapestry_compact_event_details = True
    tapestry_direction_max_samples_per_context = None
    tapestry_effect_bad_threshold = 0.001
    tapestry_effect_good_threshold = -0.001
    tapestry_enable_events = True
    tapestry_event_sampling_prob = 1.0
    tapestry_generation_prune_enabled = True
    tapestry_hash_dim = 128
    tapestry_log_only_extremes = True
    tapestry_max_generation_age = 200
    tapestry_max_graph_size = 10000
    tapestry_proj_dim = 32
    tapestry_prune_check_interval_gens = 10
    tapestry_prune_target_fraction = 0.7
    tapestry_ring_size = 32
    tapestry_snapshot_interval = 5
    torch_genome_enabled = False
    trust_radius_expand = 1.1
    trust_radius_init_fraction = 0.005
    trust_radius_shrink = 0.5
    unlock_fraction_high_dim = 0.08
    unlock_fraction_static = 0.05
    enable_pnn_lock_quota = True
    pnn_quota_fitness_scale = 1.0
    pnn_quota_min_fraction = 0.1
    pnn_quota_base_fraction = 0.3
    de_F = 0.6
    de_CR = 0.9
    pnn_time_decay_per_gen = 0.2
    pnn_improvement_eps = 1e-9
    pnn_drain_success_factor = 0.5
    pnn_drain_failure_factor = 1.5
    unlock_stress_low = 0.2
    unlock_stress_high = 0.8  
    ucb_exploration_factor = 0.5
    enable_legacy_per_child_loop = False
    enable_telemetry = True
    dimension = 30  
    DIMENSIONS = [5, 10, 30, 50]
    ENABLE_ALGOS = {
      "ABC": False,
      "AEA": True,
      "CMAES": False,
      "DE": False,
      "GA": False,
      "PSO": False,
      "Pymoo-NSGA2": False,
      "Optuna-NSGA2": False
    }
    FIXED_NFE_BUDGET = 20000
    LINEAR_NFE_PER_DIM = 50
    NUM_RUNS_PER_CASE = 3
    ANYTIME_EVAL_STEP = 50
    ONLY_DIMS = [5, 10, 30, 50]
    ONLY_PROBLEMS = ["composition1", "zakharov", "dixonprice", "happycat"]
    RUN_SANITY_CHECK = False
    FILL_TO_BUDGET_AFTER_SOLVE = True
    RANDOM_SEED_BASE = 52367144
    PER_PROBLEM_BUDGETS = {}
    EVAL_BACKEND = "cuda"
    TORCH_DEVICE = "cuda"
    TORCH_DEVICES = 0
    USE_SUBPROCESSES = True
    PARALLEL_WORKERS = 12
    PARALLEL_BACKEND = "thread"
    RUN_TIMEOUT_SEC = 600
    OUTPUT_DIR_TEMPLATE = "results_benchmark_aea_v2_new_config_{timestamp}"
    OUTPUT_DIR_BASE = "C:\\Users\\wes\\new_aea_repository_extraction-1"
    ENABLE_FILE_LOG = True
    LOG_LEVEL = "WARNING"
    CONSOLE_LOG_LEVEL = "WARNING"   

cfg= Config()








