import imp
import os

import ml_collections

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


MODEL_PATH = os.getenv("FLUX2_MODEL_PATH", "models/Flux.2")
DATASET_JSON_PATH = os.getenv(
    "FLUX2_DATA_JSON", "data/sharegpt/text_to_image.json"
)
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "outputs")
WANDB_LOG_ROOT = os.getenv("WANDB_LOG_ROOT", "logs/wandb")


def get_config(name):
    return globals()[name]()


def _configure_grouped_batching(
    config,
    n_gpus,
    num_groups,
    num_image_per_prompt,
    gradient_step_per_epoch=1,
    train_batch_size=None,
):
    total_samples = num_groups * num_image_per_prompt
    config.sample.num_image_per_prompt = num_image_per_prompt
    config.num_groups = num_groups

    if train_batch_size is not None:
        bsz = int(train_batch_size)
        if not (
            total_samples % (n_gpus * bsz) == 0
            and (n_gpus * bsz) % num_image_per_prompt == 0
        ):
            raise ValueError(
                "Invalid batch sizing for FLUX2: num_groups*num_image_per_prompt must be divisible by "
                "(n_gpus*train_batch_size), and (n_gpus*train_batch_size) must be divisible by num_image_per_prompt."
            )
    else:
        # Keep default behavior consistent with existing FLUX2 setting preference.
        bsz = 9
        while bsz >= 1:
            if (
                total_samples % (n_gpus * bsz) == 0
                and (n_gpus * bsz) % num_image_per_prompt == 0
            ):
                break
            bsz -= 1
        if bsz < 1:
            raise ValueError("Cannot find valid train_batch_size for requested num_groups/num_image_per_prompt.")

    num_batches_per_epoch = total_samples // (n_gpus * bsz)
    if num_batches_per_epoch % gradient_step_per_epoch != 0:
        raise ValueError("num_batches_per_epoch must be divisible by gradient_step_per_epoch.")

    config.sample.train_batch_size = bsz
    config.sample.num_batches_per_epoch = num_batches_per_epoch
    config.train.batch_size = bsz
    config.train.gradient_accumulation_steps = num_batches_per_epoch // gradient_step_per_epoch


def _build_base_config(exp_name):
    config = base.get_config()
    n_gpus = int(os.getenv("N_GPUS", "8"))
    num_groups = int(os.getenv("NUM_GROUPS", "48"))
    num_image_per_prompt = int(os.getenv("NUM_IMAGE_PER_PROMPT", "12"))

    config.base_model = "flux2_klein"
    config.run_name = exp_name

    config.pretrained.model = MODEL_PATH
    config.pretrained.revision = ""

    # Data: direct prompt JSON (no pre-encoded prompt_embed/text_ids).
    config.dataset_json_path = DATASET_JSON_PATH
    config.dataset_test_ratio = 0.02
    config.dataset_split_seed = 42
    config.num_workers = 4

    # Flux2 text encoder settings.
    config.max_sequence_length = 512
    config.text_encoder_out_layers = [9, 18, 27]

    # Align with sd35_sharegpt_qwenvl_flowgrpo_base hyper-parameters where applicable.
    config.resolution = 512
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.train_guidance_scale = 1.0
    config.eval_guidance_scale = 1.0
    # FLUX.2 expects guidance embedding even when CFG=1.
    config.train_embedded_guidance = 1.0
    config.eval_embedded_guidance = 1.0
    config.sample.noise_level = 0.7
    _configure_grouped_batching(
        config,
        n_gpus=n_gpus,
        num_groups=num_groups,
        num_image_per_prompt=num_image_per_prompt,
        gradient_step_per_epoch=1,
    )
    config.sample.test_batch_size = 16
    config.sample.global_std = True
    config.sample.deterministic = True
    config.sample.solver = "dpm2"

    # Train setup.
    config.use_lora = True
    config.lora = ml_collections.ConfigDict()
    config.lora.rank = 32
    config.lora.alpha = 64
    config.lora.target_modules = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.linear_in",
        "ff.linear_out",
        "ff_context.linear_in",
        "ff_context.linear_out",
        "attn.to_qkv_mlp_proj",
        "single_transformer_blocks.*.attn.to_out*",
    ]

    config.train.learning_rate = 5e-5
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0
    config.train.num_inner_epochs = 1
    # Use almost all sampled timesteps (same style as SD3.5 flowgrpo) for stabler optimization.
    config.train.timestep_fraction = 0.99
    config.train.adv_clip_max = 5
    config.train.beta = 0.001
    # FLUX.2 is sensitive to noisy negative advantages; keep only positive updates by default.
    config.train.adv_mode = "positive_only"
    config.train.policy_loss_scale = 0.5
    config.train.ema = True
    config.train.lora_path = None
    config.train.gradient_checkpointing = True

    # Damp the implicit negative branch in DiffusionNFT loss to reduce reward collapse.
    config.beta = 0.2
    config.decay_type = 1
    config.mixed_precision = "bf16"
    config.model_load_dtype = "bf16"
    config.allow_tf32 = True
    config.per_prompt_stat_tracking = True

    # Reward setup: align with SD3.5 flowgrpo reward config.
    config.reward_fn = {"qwen_vl": 1.0}
    config.reward_client_workers = 8
    config.reward_max_pending = 1

    config.num_epochs = 100000
    config.save_freq = 150
    config.eval_freq = 1000
    config.eval_steps = 1000
    config.eval_num_prompts = 64

    config.wandb_mode = "offline"
    config.output_dir = f"{OUTPUT_ROOT}/{exp_name}"
    config.log_dir = f"{WANDB_LOG_ROOT}/{exp_name}"
    config.save_dir = config.output_dir

    return config


def flux2_klein_sharegpt_qwenvl():
    exp_name = os.getenv("EXP_NAME", "flux2_klein_diffusionnft_sharegpt_8gpu")
    config = _build_base_config(exp_name=exp_name)
    return config


def flux2_klein_sharegpt_qwenvl_debug():
    config = flux2_klein_sharegpt_qwenvl()
    config.debug = True
    config.num_epochs = 1
    config.sample.num_batches_per_epoch = 2
    config.train.gradient_accumulation_steps = 1
    config.sample.num_steps = 25
    config.eval_steps = 10000
    config.save_freq = 50
    config.eval_freq = 100000
    config.wandb_mode = "disabled"
    return config
