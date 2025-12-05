import ads
from ads.model.datascience_model_group import DataScienceModelGroup
import json

# 1. Set Auth with the specific profile
ads.set_auth(auth="security_token", profile="aryan-ashburn2")

# 2. Get the Model Group created by your deployment (from latest logs)
group_id = "ocid1.datasciencemodelgroupint.oc1.iad.amaaaaaav66vvniafod77ugys4lya3xsq75frfpxzjbjbcipohli6pibik3q"

try:
    model_group = DataScienceModelGroup.from_id(group_id)

    # 3. Extract the configuration
    config_value = model_group.custom_metadata_list.get("MULTI_MODEL_CONFIG").value
    config_json = json.loads(config_value)

    # 4. Print and Verify
    print("\n--- Verification Results ---")
    for model in config_json['models']:
        print(f"\nModel Name: {model.get('model_name', 'Unknown')}")
        print(f"Params:     {model['params']}")
        
        if "--max-model-len" in model['params']:
            print(">> STATUS: Has SMM Defaults (Expected for 'Llama_Default2')")
        else:
            print(">> STATUS: Clean / No Defaults (Expected for 'Llama_Clear2')")

except Exception as e:
    print(f"Error fetching model group: {e}")

