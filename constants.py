################################################################################
# NOTE: this is an artifact file, a constants file for experiments 
#       that were run before estimated activation scaling factors were stored
#       with the checkpoint's config
################################################################################

scaling_factor_dict = {
    "blocks.3.hook_resid_pre" : {
        "EleutherAI/pythia-1b/step8": None,
        "EleutherAI/pythia-1b/step64": None,
        "EleutherAI/pythia-1b/step128": None,
        "EleutherAI/pythia-1b/step512": None,
        "EleutherAI/pythia-1b/step2000": None,
        "EleutherAI/pythia-1b/main": None,
    },
    "blocks.5.hook_resid_pre" : {
        "EleutherAI/pythia-1b/step8": [4.3378],
        "EleutherAI/pythia-1b/step64": [3.7454],
        "EleutherAI/pythia-1b/step128": [4.0344],
        "EleutherAI/pythia-1b/step512": [3.3081],
        "EleutherAI/pythia-1b/step2000": [1.5482],
        "EleutherAI/pythia-1b/main": [0.9953],
        "bigscience/bloom-1b1-intermediate/global_step1000": [1.0080],
        "bigscience/bloom-1b1-intermediate/global_step10000": [0.4790],
        "bigscience/bloom-1b1-intermediate/global_step100000": [0.3464],
        "bigscience/bloom-1b1-intermediate/main": [0.9790]
    },
    "blocks.7.hook_resid_pre" : {
        "EleutherAI/pythia-1b/step8": None,
        "EleutherAI/pythia-1b/step64": None,
        "EleutherAI/pythia-1b/step128": None,
        "EleutherAI/pythia-1b/step512": None,
        "EleutherAI/pythia-1b/step2000": None,
        "EleutherAI/pythia-1b/main": None,
        
    },
    "blocks.9.hook_resid_pre" : {
        "EleutherAI/pythia-1b/step8": [3.1020],
        "EleutherAI/pythia-1b/step64": [2.2521],
        "EleutherAI/pythia-1b/step128": [2.5277],
        "EleutherAI/pythia-1b/step512": [2.0862],
        "EleutherAI/pythia-1b/step2000": [1.1974],
        "EleutherAI/pythia-1b/main": [0.8227],
        
    },
    "blocks.10.hook_resid_pre": {
        "base": [0.2773],
        "rand": [0.1867],
        "chat": [0.2449],
        #
        "base_vs_base": [0.2773, 0.2773],  
        "base_vs_rand":  [0.2773, 0.1867], 
        "base_vs_chat": [0.2773, 0.2449], 
        "base_vs_chat_vs_rand": [0.2773, 0.2449, 0.1867], 
        #
        "EleutherAI/pythia-1b/step0": [3.0061],
        "EleutherAI/pythia-1b/step2": [3.0061],
        "EleutherAI/pythia-1b/step2000": [1.1147],
        "EleutherAI/pythia-1b/step8000": [0.5490],
        "EleutherAI/pythia-1b/step16000": [0.4020],
        "EleutherAI/pythia-1b/step110000": [0.6445],
        "EleutherAI/pythia-1b/step120000": [0.6880],
        #
        "allenai/OLMo-1B-0724-hf/main": [9.4749],
        "allenai/OLMo-1B-0724-hf/step1454000-tokens3048B": [9.4748],
        "allenai/OLMo-1B-0724-hf/step137000-tokens287B": [1.7319],
        "allenai/OLMo-1B-0724-hf/step16000-tokens33B": [0.9219],
        "allenai/OLMo-1B-0724-hf/step2000-tokens4B": [0.9680],
        "allenai/OLMo-1B-0724-hf/step0-tokens0B": [1.0973],
        "allenai/OLMo-7B-0724-hf/step68500-tokens287B": [2.9354],
        "allenai/OLMo-7B-0724-hf/step650650-tokens2729B": [12.3911],
    },
    "blocks.11.hook_resid_pre" : {
        "EleutherAI/pythia-1b/step8": None,
        "EleutherAI/pythia-1b/step64": None,
        "EleutherAI/pythia-1b/step128": None,
        "EleutherAI/pythia-1b/step512": None,
        "EleutherAI/pythia-1b/step2000": None,
        "EleutherAI/pythia-1b/main": None,
    },
    "blocks.13.hook_resid_pre": {
        "bigscience/bloom-1b1-intermediate/global_step1000": [0.7808],
        "bigscience/bloom-1b1-intermediate/global_step10000": [0.3841],
        "bigscience/bloom-1b1-intermediate/global_step100000": [0.2687],
        "bigscience/bloom-1b1-intermediate/main": [0.7489]
    },
    "blocks.15.hook_resid_pre": {
        "bigscience/bloom-1b1-intermediate/global_step1000": [0.7649],
        "bigscience/bloom-1b1-intermediate/global_step10000": [0.3612],
        "bigscience/bloom-1b1-intermediate/global_step100000": [0.2520],
        "bigscience/bloom-1b1-intermediate/main": [0.6904]
    }
}
