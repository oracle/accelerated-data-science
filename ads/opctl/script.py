import argparse
import json

from ads.common.auth import AuthContext
from ads.model.generic_model import GenericModel


def verify(artifact_dir, payload, auth, profile): # pragma: no cover
    kwargs = {"auth": auth}
    if profile != 'None':
        kwargs["profile"] = profile
    with AuthContext(**kwargs):
        model = GenericModel.from_model_artifact(
            uri=artifact_dir,
            artifact_dir=artifact_dir,
            force_overwrite=True,
        )

        try:
            payload = json.loads(payload)
        except:
            pass
        print(model.verify(payload, auto_serialize_data=False))


def main(): # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    parser.add_argument("--artifact-directory", type=str, required=True)
    parser.add_argument("--auth", type=str, required=True)
    parser.add_argument("--profile", type=str,required=False)
    args = parser.parse_args()
    verify(
        artifact_dir=args.artifact_directory, payload=args.payload, auth=args.auth, profile=args.profile
    )
    return 0


if __name__ == "__main__": # pragma: no cover
    main()
