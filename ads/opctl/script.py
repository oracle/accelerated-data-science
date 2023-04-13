import json
import sys
import tempfile

from ads.model.generic_model import GenericModel


def verify(artifact_dir, data, compartment_id, project_id):
    with tempfile.TemporaryDirectory() as td:
        model = GenericModel.from_model_artifact(uri = artifact_dir, artifact_dir=artifact_dir,
                                                force_overwrite=True,
                                                compartment_id=compartment_id,
                                                project_id=project_id)

        data = json.loads(data)
        print(model.verify(data, auto_serialize_data=False))


def main():
    args = sys.argv[1:]
    verify(artifact_dir = args[0], data=args[1], compartment_id=args[2], project_id=args[3])
    return 0


if __name__ == "__main__":
    main()
