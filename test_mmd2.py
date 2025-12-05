import ads
from ads.model.datascience_model_group import DataScienceModelGroup
import json

# 1. Set Auth with the specific profile
ads.set_auth(auth="security_token", profile="aryan-ashburn2")

# 2. Get the Model Group created by your deployment (from latest logs)
group_id = "ocid1.datasciencemodelgroupint.oc1.iad.amaaaaaav66vvniazm2a2ao2u7n65baecxtu6e6lejfvj7gb3ytu3zduq35q"

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
        
        if "--max-model-len 1024" in model['params']:
            print(">> STATUS: SUCCESS - Custom value used (1024)")
        elif "--max-model-len 65536" in model['params']:
            print(">> STATUS: FAIL - Defaults merged in (65536)")
        else:
            print(">> STATUS: FAIL - Param missing entirely")

except Exception as e:
    print(f"Error fetching model group: {e}")
