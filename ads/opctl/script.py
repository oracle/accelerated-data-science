import json
import sys

from ads.model.generic_model import GenericModel


def verify(artifact_dir, data): # pragma: no cover

    model = GenericModel.from_model_artifact(
        uri=artifact_dir,
        artifact_dir=artifact_dir,
        force_overwrite=True,
    )

    try:
        data = json.loads(data)
    except:
        pass
    print(model.verify(data, auto_serialize_data=False))


def main(): # pragma: no cover
    args = sys.argv[1:]
    verify(
        artifact_dir=args[0], data=args[1]
    )
    return 0


if __name__ == "__main__": # pragma: no cover
    main()
