import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

SHAREGPT_FILTERED_JSON = os.getenv("TRAIN_DATA_JSON", "data/sharegpt/text_to_image.json")
NFT_LOG_ROOT = os.getenv("NFT_LOG_ROOT", "logs/nft")


def get_config(name):
    return globals()[name]()


def _get_config(
    base_model="sd3",
    n_gpus=1,
    gradient_step_per_epoch=1,
    dataset="pickscore",
    reward_fn={},
    name="",
    num_groups=48,
    num_image_per_prompt=24,
    train_batch_size=None,
):
    config = base.get_config()
    assert base_model in ["sd3"]
    assert dataset in ["pickscore", "ocr", "geneval", "sharegpt"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 9

    config.sample.num_image_per_prompt = num_image_per_prompt

    if train_batch_size is not None:
        bsz = train_batch_size
        if not (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            raise ValueError(
                "Invalid batch sizing: num_groups*num_image_per_prompt must be divisible by "
                "(n_gpus*train_batch_size), and (n_gpus*train_batch_size) must be divisible by "
                "num_image_per_prompt."
            )
        n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
        if n_batch_per_epoch % gradient_step_per_epoch != 0:
            raise ValueError(
                "Invalid batch sizing: num_batches_per_epoch must be divisible by gradient_step_per_epoch."
            )
        config.sample.train_batch_size = bsz
        config.sample.num_batches_per_epoch = n_batch_per_epoch
        config.train.batch_size = config.sample.train_batch_size
        config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // gradient_step_per_epoch
    else:
        while True:
            if bsz < 1:
                assert False, "Cannot find a proper batch size."
            if (
                num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
                and bsz * n_gpus % config.sample.num_image_per_prompt == 0
            ):
                n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
                if n_batch_per_epoch % gradient_step_per_epoch == 0:
                    config.sample.train_batch_size = bsz
                    config.sample.num_batches_per_epoch = n_batch_per_epoch
                    config.train.batch_size = config.sample.train_batch_size
                    config.train.gradient_accumulation_steps = (
                        config.sample.num_batches_per_epoch // gradient_step_per_epoch
                    )
                    break
            bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    if dataset == "geneval":
        config.prompt_fn = "geneval"
    elif dataset == "sharegpt":
        config.prompt_fn = "sharegpt"
    else:
        config.prompt_fn = "general_ocr"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def sd3_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=2, dataset="ocr", reward_fn=reward_fn, name="ocr"
    )
    config.beta = 0.1
    config.decay_type = 2
    return config


def sd3_geneval():
    reward_fn = {
        "geneval": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )
    return config


def sd3_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    return config


def sd3_hpsv2():
    reward_fn = {
        "hpsv2": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=1, dataset="pickscore", reward_fn=reward_fn, name="hpsv2"
    )
    return config


def sd3_multi_reward():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    return config


def sd3_sharegpt_qwenvl():
    reward_fn = {
        "qwen_vl": 1.0,
    }
    n_gpus = int(os.getenv("N_GPUS", "8"))
    config = _get_config(
        base_model="sd3",
        n_gpus=n_gpus,
        gradient_step_per_epoch=1,
        dataset="sharegpt",
        reward_fn=reward_fn,
        name="sharegpt_qwenvl",
    )
    config.dataset = os.getenv("SHAREGPT_DATA_ROOT", "data/sharegpt")
    config.pretrained.model = os.getenv("SD3_MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium")
    return config


def sd35_sharegpt_qwenvl_flowgrpo_reward():
    reward_fn = {
        "qwen_vl": 1.0,
    }
    n_gpus = int(os.getenv("N_GPUS", "8"))
    config = _get_config(
        base_model="sd3",
        n_gpus=n_gpus,
        gradient_step_per_epoch=1,
        dataset="sharegpt",
        reward_fn=reward_fn,
        name="sd35_sharegpt_qwenvl_flowgrpo",
        num_groups=48,
        num_image_per_prompt=12,
    )
    exp_name = os.getenv("EXP_NAME", "sd35_sharegpt_qwenvl_flowgrpo_reward")
    dataset_json = os.getenv("TRAIN_DATA_JSON", SHAREGPT_FILTERED_JSON)
    log_root = os.getenv("NFT_LOG_ROOT", NFT_LOG_ROOT)
    config.dataset = dataset_json
    config.dataset_json_path = dataset_json
    config.pretrained.model = os.getenv("SD35_MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium")
    # Explicitly align with the requested training setup
    config.sample.num_steps = 10
    config.sample.solver = "dpm2"
    config.train.learning_rate = 3e-4
    config.log_dir = os.path.join(log_root, exp_name)
    config.save_dir = config.log_dir
    return config

def sd35_sharegpt_qwenvl_flowgrpo_base():
    reward_fn = {
        "qwen_vl": 1.0,
    }
    n_gpus = int(os.getenv("N_GPUS", "8"))
    config = _get_config(
        base_model="sd3",
        n_gpus=n_gpus,
        gradient_step_per_epoch=1,
        dataset="sharegpt",
        reward_fn=reward_fn,
        name="sd35_sharegpt_qwenvl_flowgrpo",
        num_groups=48,
        num_image_per_prompt=12,
    )
    exp_name = os.getenv("EXP_NAME", "sd35_sharegpt_qwenvl_flowgrpo_base")
    dataset_json = os.getenv("TRAIN_DATA_JSON", SHAREGPT_FILTERED_JSON)
    log_root = os.getenv("NFT_LOG_ROOT", NFT_LOG_ROOT)
    config.dataset = dataset_json
    config.dataset_json_path = dataset_json
    config.pretrained.model = os.getenv("SD35_MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium")
    # Explicitly align with the requested training setup
    config.sample.num_steps = 10
    config.sample.solver = "dpm2"
    config.train.learning_rate = 3e-4
    config.log_dir = os.path.join(log_root, exp_name)
    config.save_dir = config.log_dir
    return config
